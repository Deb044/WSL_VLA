"""Job graph of the full Methodology 1 study on top of the corrected pipeline.

The zoo, evidence, fold archives, paired alignment training, diagnostics, and
generated-policy rollouts come from the manifest-driven plan in
:mod:`wsl_vla.experiments.zoo_jobs`. This module only appends the downstream
study - learning gates, gamma validation, continual learning, OOD adaptation,
mechanistic probes, reports, and the completion audit - pointed at the corrected
experiment layout.

Two rules replace the earlier plan:

* downstream stages wait for the *integrity* of the paired alignment
  diagnostics. Whether alignment helps is a recorded finding; it is never a
  precondition for running an experiment;
* no count, seed, fold, or validation index is written here. They come from the
  experiment manifest.

The downstream commands were re-pointed during the pipeline repair but have not
been executed against corrected artifacts. Treat the first run of each
downstream stage as an integration check.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .manifest import ExperimentManifest, ExperimentPaths
from .protocol import REQUIRED_SUITES
from .study_audit import CONDITIONS
from .zoo_jobs import JOB_SCHEMA_VERSION, build_experiment_job_plan, validate_job_plan


DOWNSTREAM_STAGES = (
    "downstream_gates",
    "ood_bank",
    "gamma_validation",
    "gamma_selection",
    "pilot",
    "continual",
    "ood",
    "mechanistic_probes",
    "reporting",
)
PILOT_SUITE = "libero_spatial"


def study_root(paths: ExperimentPaths) -> Path:
    """Directory of the downstream study inside the corrected experiment."""

    return paths.root / "methodology1"


def leave_one_suite_out_folds(manifest: ExperimentManifest) -> dict[str, object]:
    """``held-out suite -> fold``; the study needs exactly one fold per suite."""

    folds = {fold.held_out_suite: fold for fold in manifest.folds if fold.held_out_suite}
    missing = [suite for suite in REQUIRED_SUITES if suite not in folds]
    if missing or len(folds) != len(manifest.folds):
        raise ValueError(
            "the Methodology 1 study needs one leave-one-suite-out fold per reference suite; "
            f"experiment {manifest.name!r} lacks folds holding out {missing}"
        )
    return folds


def build_methodology1_job_plan(
    manifest: ExperimentManifest,
    *,
    manifest_path: str | Path,
    research_root: str | Path,
    gamma_candidates: Mapping[str, object],
    python: str = "python",
    data_root: str | Path | None = None,
) -> tuple[dict, ...]:
    """Corrected core pipeline followed by the downstream study."""

    if not manifest.zoo.frozen:
        raise ValueError(
            "the downstream study consumes held-out results and therefore needs a frozen zoo "
            "budget; validate the budget and run scripts/reporting/freeze_zoo_budget.py first"
        )
    folds = leave_one_suite_out_folds(manifest)
    paths = ExperimentPaths(Path(research_root), manifest.name)
    jobs = list(
        build_experiment_job_plan(
            manifest,
            manifest_path=manifest_path,
            research_root=research_root,
            python=python,
            data_root=data_root,
        )
    )
    study = study_root(paths)
    publication = study / "publication"
    data = [] if data_root is None else ["--data-root", str(data_root)]
    seeds = tuple(manifest.seeds)
    # One alignment run per fold feeds the downstream study: the first seed.
    alignment_seed = manifest.alignment.seeds[0]

    # Stages whose jobs train adapters (and then roll them out).
    training_stages = {"downstream_gates", "gamma_validation", "pilot", "continual", "ood"}

    def add(job_id, stage, command, dependencies=(), outputs=(), gpu=True, simulator=False):
        jobs.append(
            {
                "schema_version": JOB_SCHEMA_VERSION,
                "job_id": job_id,
                "stage": stage,
                "command": [str(value) for value in command],
                "dependencies": sorted(set(dependencies)),
                "expected_outputs": [str(Path(value)) for value in outputs],
                "resources": {
                    "gpu_required": bool(gpu),
                    "simulator_required": bool(simulator),
                    "gpu_training": stage in training_stages or job_id.startswith("m1:recovery:"),
                    "deterministic_gpu": True,
                },
                "environment": {},
            }
        )

    def checkpoints(suite: str) -> tuple[Path, Path, str]:
        fold = folds[suite]
        return (
            paths.alignment_checkpoint(fold.name, alignment_seed, "aligned"),
            paths.alignment_checkpoint(fold.name, alignment_seed, "reconstruction_only"),
            # Integrity of the paired comparison, whatever its scientific outcome.
            f"diagnostics:validation:{fold.name}:seed-{alignment_seed}",
        )

    orientation = [f"gate:camera_orientation:{suite}" for suite in REQUIRED_SUITES]
    add(
        "m1:one_task_gate",
        "downstream_gates",
        [python, "scripts/eval/run_one_task_learning_gate.py", "--suite", PILOT_SUITE,
         "--task-index", 0, "--seed", seeds[0], "--output-root", study / "gates/one_task", *data],
        dependencies=["gate:octo", f"gate:camera_orientation:{PILOT_SUITE}"],
        simulator=True,
    )
    add(
        "m1:two_task_gate",
        "downstream_gates",
        [python, "scripts/eval/run_official_continual.py", "--suite", PILOT_SUITE,
         "--condition", "sequential_no_regularization", "--seed", seeds[0], "--task-count", 2,
         "--output-root", study / "gates/continual", *data],
        dependencies=["m1:one_task_gate"],
        simulator=True,
    )

    bank_ids = {}
    selection_ids = {}
    patience_values = tuple(int(value) for value in gamma_candidates["early_stopping_patience"])
    for held_out in REQUIRED_SUITES:
        aligned, _, integrity = checkpoints(held_out)
        fold_root = study / "folds" / held_out
        bank_id = f"m1:ood_bank:{held_out}"
        add(
            bank_id,
            "ood_bank",
            [python, "scripts/data/build_ood_reference_bank.py", paths.population, aligned,
             "--held-out-suite", held_out, "--output", fold_root / "ood_reference_bank.npz"],
            dependencies=[integrity],
            outputs=[fold_root / "ood_reference_bank.npz"],
            gpu=False,
        )
        bank_ids[held_out] = bank_id

        gamma_ids = []
        for family in ("proposed", "uniform"):
            for candidate in gamma_candidates[family]:
                for patience in patience_values:
                    for source in (suite for suite in REQUIRED_SUITES if suite != held_out):
                        for seed in seeds:
                            job_id = (
                                f"m1:gamma:{held_out}:{source}:seed-{seed}:{family}:"
                                f"{candidate['name']}:p-{patience}"
                            )
                            add(
                                job_id,
                                "gamma_validation",
                                [python, "scripts/eval/run_gamma_validation.py",
                                 "--held-out-suite", held_out, "--source-suite", source,
                                 "--seed", seed, "--family", family,
                                 "--gamma-vision", candidate["vision"],
                                 "--gamma-language", candidate["language"],
                                 "--gamma-action", candidate["action"],
                                 "--patience", patience, "--alignment-checkpoint", aligned,
                                 "--evidence-root", paths.evidence,
                                 "--output-root", study / "gamma_validation", *data],
                                dependencies=[integrity, "m1:two_task_gate", *orientation],
                                simulator=True,
                            )
                            gamma_ids.append(job_id)
        selection_id = f"m1:gamma_select:{held_out}"
        add(
            selection_id,
            "gamma_selection",
            [python, "scripts/eval/select_validation_gammas.py",
             study / "gamma_validation" / held_out, "--held-out-suite", held_out,
             "--seeds", ",".join(str(seed) for seed in seeds),
             "--output", fold_root / "gamma_selection.json",
             "--report", fold_root / "gamma_selection.report.json"],
            dependencies=gamma_ids,
            outputs=[fold_root / "gamma_selection.json", fold_root / "gamma_selection.report.json"],
            gpu=False,
        )
        selection_ids[held_out] = selection_id

    def continual_command(suite, seed, condition, tier, output_root):
        aligned, control, _ = checkpoints(suite)
        return [
            python, "scripts/eval/run_official_continual.py", "--suite", suite, "--seed", seed,
            "--condition", condition,
            "--gamma-selection", study / "folds" / suite / "gamma_selection.json",
            "--alignment-checkpoint", aligned, "--reconstruction-checkpoint", control,
            "--evidence-root", paths.evidence, "--tier", tier, "--output-root", output_root, *data,
        ]

    pilot_ids = []
    for seed in seeds:
        for condition in CONDITIONS:
            job_id = f"m1:pilot:{PILOT_SUITE}:seed-{seed}:{condition}"
            add(
                job_id,
                "pilot",
                continual_command(
                    PILOT_SUITE, seed, condition, "development", study / "development/continual"
                ),
                dependencies=[selection_ids[PILOT_SUITE]],
                simulator=True,
            )
            pilot_ids.append(job_id)

    continual_ids = []
    ood_ids = []
    swap_ids = []
    recovery_ids = []
    for suite in REQUIRED_SUITES:
        aligned, _, _ = checkpoints(suite)
        for seed in seeds:
            for condition in CONDITIONS:
                job_id = f"m1:continual:{suite}:seed-{seed}:{condition}"
                add(
                    job_id,
                    "continual",
                    continual_command(
                        suite, seed, condition, "publication", publication / "continual"
                    ),
                    dependencies=[selection_ids[suite], *pilot_ids],
                    simulator=True,
                )
                continual_ids.append(job_id)
            ood_id = f"m1:ood:{suite}:seed-{seed}"
            add(
                ood_id,
                "ood",
                [python, "scripts/eval/run_official_ood.py", "--suite", suite, "--seed", seed,
                 "--alignment-checkpoint", aligned,
                 "--reference-bank", study / "folds" / suite / "ood_reference_bank.npz",
                 "--evidence-root", paths.evidence, "--tier", "publication",
                 "--output-root", publication / "ood", *data],
                dependencies=[bank_ids[suite], *orientation],
                simulator=True,
            )
            ood_ids.append(ood_id)
            source = publication / "continual/ten_task_study" / suite / f"seed_{seed}"
            swap_id = f"m1:component_swap:{suite}:seed-{seed}"
            add(
                swap_id,
                "mechanistic_probes",
                [python, "scripts/eval/run_component_swaps.py", source / "proposed_asymmetric",
                 "--tier", "publication", "--output-root", publication / "component_swaps", *data],
                dependencies=[f"m1:continual:{suite}:seed-{seed}:proposed_asymmetric"],
                simulator=True,
            )
            swap_ids.append(swap_id)
            recovery_id = f"m1:recovery:{suite}:seed-{seed}"
            add(
                recovery_id,
                "mechanistic_probes",
                [python, "scripts/eval/run_recovery_probe.py",
                 source / "sequential_no_regularization",
                 "--output-root", publication / "recovery", *data],
                dependencies=[f"m1:continual:{suite}:seed-{seed}:sequential_no_regularization"],
                simulator=True,
            )
            recovery_ids.append(recovery_id)

    reports = publication / "reports"
    add(
        "m1:report:continual",
        "reporting",
        [python, "scripts/reporting/report_publication_study.py",
         publication / "continual/ten_task_study",
         "--output-json", reports / "continual_study.json",
         "--output-parquet", reports / "continual_records.parquet"],
        dependencies=continual_ids,
        gpu=False,
    )
    add(
        "m1:report:ood",
        "reporting",
        [python, "scripts/reporting/report_ood_study.py", publication / "ood",
         "--output-json", reports / "ood_study.json",
         "--output-parquet", reports / "ood_records.parquet"],
        dependencies=ood_ids,
        gpu=False,
    )
    add(
        "m1:report:components",
        "reporting",
        [python, "scripts/reporting/report_component_swaps.py", publication / "component_swaps",
         "--output-json", reports / "component_swaps.json",
         "--output-parquet", reports / "component_swaps.parquet",
         "--output-figure", reports / "component_drift_scatter.png"],
        dependencies=swap_ids,
        gpu=False,
    )
    add(
        "m1:audit",
        "reporting",
        [python, "scripts/reporting/audit_methodology1_study.py", "--manifest", manifest_path,
         "--research-root", research_root, "--output", reports / "completion_audit.json"],
        dependencies=["m1:report:continual", "m1:report:ood", "m1:report:components",
                      "report:rollouts", *recovery_ids],
        outputs=[reports / "completion_audit.json"],
        gpu=False,
    )
    validate_job_plan(jobs)
    return tuple(jobs)


def summarize_job_plan(jobs: Sequence[Mapping[str, object]]) -> dict:
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_count": len(jobs),
        "gpu_job_count": sum(bool(job["resources"]["gpu_required"]) for job in jobs),
        "simulator_job_count": sum(bool(job["resources"]["simulator_required"]) for job in jobs),
        "jobs_by_stage": dict(sorted(Counter(str(job["stage"]) for job in jobs).items())),
        "downstream_status": (
            "downstream stages (m1:*) were re-pointed at the corrected pipeline but have not "
            "been executed against corrected artifacts"
        ),
    }
