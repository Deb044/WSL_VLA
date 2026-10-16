#!/usr/bin/env python3
"""Verify rollout camera orientation against benchmark demonstration generation.

Two independent checks, both recorded in ``camera_orientation.json``:

1. **Declared convention.** Every LIBERO demonstration file records the
   robosuite image convention it was generated with
   (``data.attrs.macros_image_convention``) and its frame size. Both must match
   the shared preprocessing configuration.
2. **Measured alignment.** Recorded simulator states of real demonstrations are
   restored in the official benchmark environment, re-rendered, and compared
   with the stored frames under every candidate transform. The transform with
   the smallest error must be the configured ``simulator_alignment``, by a clear
   margin.

It also saves, for human inspection, the stored frame, the re-rendered frame,
and the frame exactly as the checkpoint will see it (after ``image_transform``).
Check 1 needs only the data files; check 2 needs the simulator. The report's
``status`` is ``verified`` only when both ran and agree. Without the simulator
the command exits 3 (``simulator_unavailable``) unless ``--declared-only`` was
requested, in which case the status is ``declared_only`` - never ``verified``.

Exit codes: 0 verified (or the requested declared-only check passed), 2 a check
failed, 3 the simulator check could not run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.data.libero import StrictLiberoHDF5
from wsl_vla.data.preprocessing import (
    IMAGE_TRANSFORMS,
    LIBERO_CAMERA_SOURCES,
    apply_image_transform,
)
from wsl_vla.evaluation.libero_rollout import benchmark_task_indices, robosuite_import_environment
from wsl_vla.experiments.manifest import task_id as make_task_id
from wsl_vla.experiments.protocol import load_yaml, validate_reference_tasks, validate_research_config
from wsl_vla.experiments.runtime import task_data_file

MARGIN = 2.0  # the best transform's error must be at least this factor below the runner-up


def measured_alignment(simulator_frame: np.ndarray, stored_frame: np.ndarray) -> dict:
    errors = {
        name: float(
            np.mean(
                np.square(
                    apply_image_transform(simulator_frame[None], name)[0].astype(np.float64)
                    - stored_frame.astype(np.float64)
                )
            )
        )
        for name in IMAGE_TRANSFORMS
    }
    ranked = sorted(errors, key=errors.get)
    return {
        "mean_squared_error": errors,
        "best": ranked[0],
        "margin": errors[ranked[1]] / max(errors[ranked[0]], 1e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--declared-only",
        action="store_true",
        help="run only the data-file check (the simulator check is then reported as not run)",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    settings = config["preprocessing"]
    source = (int(settings["source_camera_height"]), int(settings["source_camera_width"]))
    identifier = make_task_id(args.suite, args.task_index)
    data_file, _ = task_data_file(args.data_root or config["data"]["root"], tasks, identifier)
    dataset = StrictLiberoHDF5(data_file, require_wrist_camera=True)
    camera_keys = tuple(value[0] for value in LIBERO_CAMERA_SOURCES.values())
    episodes = list(
        dataset.iter_episodes(
            dataset.episode_ids[: args.episodes], observation_keys=camera_keys, include_states=True
        )
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    problems = []
    declared = {
        "data_file": str(data_file),
        "recorded_image_convention": dataset.image_convention,
        "configured_image_convention": settings["demonstration_image_convention"],
        "frame_sizes": {
            key: list(np.asarray(episodes[0].observations[key]).shape[1:3]) for key in camera_keys
        },
        "configured_source_resolution": list(source),
    }
    if declared["recorded_image_convention"] != declared["configured_image_convention"]:
        problems.append("demonstration image convention differs from the preprocessing config")
    if any(tuple(size) != source for size in declared["frame_sizes"].values()):
        problems.append("demonstration frame size differs from preprocessing.source_camera_*")

    measured: dict = {"status": "not_run", "reason": "--declared-only"}
    if not args.declared_only:
        try:
            with robosuite_import_environment():
                import robosuite.macros as macros
                from libero.libero import benchmark, get_libero_path
                from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            measured = {"status": "not_run", "reason": f"LIBERO simulator unavailable: {exc}"}
        else:
            suite = benchmark.get_benchmark_dict()[args.suite]()
            # The protocol's task order differs from LIBERO's in some suites.
            benchmark_index = benchmark_task_indices(suite, tasks["suites"][args.suite])[args.task_index]
            task = suite.get_task(benchmark_index)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), camera_heights=source[0], camera_widths=source[1]
            )
            comparisons = []
            try:
                env.seed(0)
                env.reset()
                for episode in episodes:
                    observation = env.set_init_state(episode.states[0])
                    for octo_key, (dataset_key, simulator_key) in LIBERO_CAMERA_SOURCES.items():
                        rendered = np.asarray(observation[simulator_key])
                        stored = np.asarray(episode.observations[dataset_key][0])
                        result = measured_alignment(rendered, stored)
                        comparisons.append(
                            {"episode": episode.episode_id, "camera": octo_key, **result}
                        )
                        if episode is episodes[0]:
                            _save_triptych(output, octo_key, stored, rendered, settings)
            finally:
                env.close()
            best = {item["best"] for item in comparisons}
            measured = {
                "status": "run",
                "benchmark_task_index": benchmark_index,
                "benchmark_bddl_file": task.bddl_file,
                "simulator_image_convention": str(macros.IMAGE_CONVENTION),
                "configured_simulator_alignment": settings["simulator_alignment"],
                "comparisons": comparisons,
                "measured_alignment": sorted(best),
                "minimum_margin": min(item["margin"] for item in comparisons),
            }
            if best != {settings["simulator_alignment"]}:
                problems.append(
                    f"measured simulator alignment {sorted(best)} differs from the configured "
                    f"{settings['simulator_alignment']!r}"
                )
            if measured["minimum_margin"] < MARGIN:
                problems.append("the best orientation is not clearly better than the runner-up")

    upright = (settings["demonstration_image_convention"] == "opengl") == (
        settings["image_transform"] in {"flip_vertical", "rotate_180"}
    )
    if problems:
        status = "failed"
    elif measured["status"] == "run":
        status = "verified"
    else:
        status = "declared_only" if args.declared_only else "simulator_unavailable"
    report = {
        "schema_version": 2,
        "task_id": identifier,
        "status": status,
        "verified": status == "verified",
        "problems": problems,
        "declared_convention_check": declared,
        "measured_alignment_check": measured,
        "image_transform": settings["image_transform"],
        "checkpoint_sees_upright_frames": upright,
        "note": (
            "opengl-convention frames store rows bottom-up; 'flip_vertical' presents them "
            "upright. 'rotate_180' would additionally mirror left and right."
        ),
    }
    (output / "camera_orientation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if problems:
        return 2
    if measured["status"] != "run":
        print(
            "SIMULATOR ALIGNMENT CHECK NOT RUN - rollout camera orientation is unverified: "
            f"{measured['reason']}",
            file=sys.stderr,
        )
        return 0 if args.declared_only else 3
    return 0


def _save_triptych(output: Path, camera: str, stored, rendered, settings) -> None:
    from PIL import Image

    aligned = apply_image_transform(rendered[None], settings["simulator_alignment"])[0]
    presented = apply_image_transform(stored[None], settings["image_transform"])[0]
    panel = np.concatenate([stored, aligned, presented], axis=1)
    Image.fromarray(np.ascontiguousarray(panel)).save(
        output / f"{camera}_stored__simulator_aligned__presented_to_checkpoint.png"
    )


if __name__ == "__main__":
    raise SystemExit(main())
