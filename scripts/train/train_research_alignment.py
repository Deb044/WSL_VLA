#!/usr/bin/env python3
"""Train one primary alignment condition on a fold archive.

Run this twice per fold and seed - once with ``--condition aligned`` and once
with ``--condition reconstruction_only``. The two runs share architecture,
initialization, batch sequence, optimizer exposure, evidence classification,
and reconstruction-based checkpoint selection; only the contrastive terms
differ. Command-line overrides take precedence over the configuration and the
manifest, and the fully resolved configuration is stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.alignment.archive import open_alignment_archive
from wsl_vla.alignment.checkpoint import (
    load_alignment_checkpoint,
    matching_checkpoint_metadata,
    save_alignment_checkpoint,
)
from wsl_vla.alignment.training import (
    INTEGER_KEYS,
    OVERRIDABLE_KEYS,
    fit_alignment_geometry,
    resolve_alignment_config,
    train_alignment,
)
from wsl_vla.experiments.manifest import ALIGNMENT_CONDITIONS
from wsl_vla.experiments.provenance import git_state
from wsl_vla.experiments.runtime import (
    add_experiment_arguments,
    environment_record,
    input_hashes,
    load_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser, checkpoint=False)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True, choices=ALIGNMENT_CONDITIONS)
    # Every override defaults to "not given"; an absent flag never shadows the
    # configured value.
    for key in OVERRIDABLE_KEYS:
        parser.add_argument(
            f"--{key.replace('_', '-')}",
            type=int if key in INTEGER_KEYS else float,
            default=None,
            help="override the configured value",
        )
    args = parser.parse_args()

    config, _, manifest, paths = load_experiment(args)
    fold = manifest.fold(args.fold)
    output = paths.alignment_checkpoint(fold.name, args.seed, args.condition)
    environment = environment_record()
    archive = open_alignment_archive(paths.archive(fold.name))
    if archive.manifest["experiment_manifest_sha256"] != manifest.sha256():
        raise ValueError("the fold archive was built for a different experiment manifest")
    resolved = resolve_alignment_config(
        alignment_config=config["alignment"],
        manifest=manifest,
        archive=archive,
        condition=args.condition,
        seed=args.seed,
        overrides={key: getattr(args, key) for key in OVERRIDABLE_KEYS},
    )
    finished = matching_checkpoint_metadata(output, config=resolved, archive=archive)
    if finished is not None:
        load_alignment_checkpoint(output)  # still verifies every file hash
        print(json.dumps({"ok": True, "already_complete": str(output)}, indent=2))
        return 0
    training = train_alignment(resolved, archive)
    objective = training["objective"]
    geometry = fit_alignment_geometry(
        objective, training["params"], archive, batch_size=int(resolved["micro_batch_size"])
    )
    commit, dirty = git_state(REPO_ROOT)
    metadata = save_alignment_checkpoint(
        output,
        config=resolved,
        params=training["params"],
        archive=archive,
        entry_scales=objective.entry_scales,
        geometry=geometry,
        training=training,
        provenance={
            "command": list(sys.argv),
            "git_commit": commit,
            "git_dirty": dirty,
            "environment": environment,
            "inputs": input_hashes(args),
        },
    )
    # A checkpoint that cannot be reloaded is not a result.
    reloaded = load_alignment_checkpoint(output)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "condition": reloaded.condition,
                "selected_update": metadata["selected_update"],
                "best_validation_reconstruction": metadata["best_validation_reconstruction"],
                "initial_params_sha256": metadata["initial_params_sha256"],
                "batch_schedule_sha256": metadata["batch_schedule_sha256"],
                "mapper_selection": {
                    name: value["selected_ridge_alpha"]
                    for name, value in metadata["mapper_selection"].items()
                },
                "seconds_per_optimizer_update": metadata["resources"]["seconds_per_optimizer_update"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
