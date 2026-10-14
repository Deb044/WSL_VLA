"""Machine-checkable completion audit for the Methodology 1 study.

The audit reads the corrected experiment layout
(``research_root/experiments/<name>``) and the downstream study inside it
(``.../methodology1``). Every expected count comes from the experiment manifest.

It checks that work is *complete and internally consistent*. It never checks
that a method won: paired alignment diagnostics must pass integrity, and their
scientific outcome (supports, inconclusive, negative) is reported as recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .manifest import ALIGNMENT_CONDITIONS, ExperimentManifest, ExperimentPaths, split_task_id
from .protocol import REQUIRED_SUITES


CONDITIONS = (
    "sequential_no_regularization",
    "replay_10",
    "replay_100",
    "uniform_regularization",
    "proposed_asymmetric",
    "direction_inverted",
    "reconstruction_only_latent",
    "independent_adapter_oracle",
)
CHECKPOINT_FILES = (
    "alignment_params.msgpack",
    "metadata.json",
    "layout.npz",
    "entry_scales.npy",
    "standardization.npz",
    "task_bank.npz",
    "mappers.npz",
    "shells.npz",
)
PILOT_SUITE = "libero_spatial"
TASKS_PER_SUITE = 10


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _field(path: Path, *keys: str) -> Any:
    """Nested JSON field, or ``None`` when the file or the field is unavailable."""

    try:
        value: Any = _json(path)
        for key in keys:
            value = value[key]
        return value
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def audit_methodology1_study(
    research_root: str | Path, manifest: ExperimentManifest
) -> dict[str, Any]:
    """Audit expected artifacts without importing JAX, Octo, or LIBERO."""

    paths = ExperimentPaths(Path(research_root), manifest.name)
    root = paths.root
    study = root / "methodology1"
    publication = study / "publication"
    seeds = tuple(manifest.seeds)
    counts = manifest.derived_counts()
    checks: list[dict[str, Any]] = []
    commits: set[str] = set()
    base_hashes: set[str] = set()
    alignment_outcomes: dict[str, Any] = {}

    def add(name: str, problems: Iterable[str], **details: Any) -> None:
        items = list(problems)
        checks.append({"name": name, "passed": not items, "problems": items[:20], **details})

    # ------------------------------------------------------------ core gates
    gates = root / "gates"
    for name, path in (
        ("preflight", gates / "preflight" / "preflight.json"),
        ("zero_adapter_equivalence_and_gradient_gates", gates / "octo_gates.json"),
    ):
        add(name, [] if _field(path, "ok") is True else [f"missing or not ok: {path}"], path=str(path))
    suites = sorted({split_task_id(identifier)[0] for identifier in manifest.tasks})
    add(
        "camera_orientation_verified",
        [
            f"rollout camera orientation is not verified for {suite}"
            for suite in suites
            if _field(gates / "camera_orientation" / suite / "camera_orientation.json", "verified")
            is not True
        ],
        expected=len(suites),
    )

    # ------------------------------------------------------------------ zoo
    verification = paths.population / "verification.json"
    problems = []
    if _field(verification, "ok") is not True:
        problems.append(f"missing or not ok: {verification}")
    else:
        payload = _json(verification)
        if payload.get("manifest_sha256") != manifest.sha256():
            problems.append("population was verified against a different manifest")
        if payload.get("present_samples") != counts["adapter_samples"]:
            problems.append(
                f"population holds {payload.get('present_samples')} samples, "
                f"the manifest expects {counts['adapter_samples']}"
            )
        if payload.get("incomplete_or_failed_runs"):
            problems.append("the zoo contains incomplete or failed runs")
    add("population_verification", problems, expected_samples=counts["adapter_samples"])

    # ------------------------------------------------------------ alignment
    fold_problems = []
    for fold in manifest.folds:
        archive = paths.archive(fold.name) / "archive_manifest.json"
        if not archive.is_file():
            fold_problems.append(f"missing {archive}")
        for seed in manifest.alignment.seeds:
            for condition in ALIGNMENT_CONDITIONS:
                checkpoint = paths.alignment_checkpoint(fold.name, seed, condition)
                fold_problems.extend(
                    f"missing {checkpoint / name}"
                    for name in CHECKPOINT_FILES
                    if not (checkpoint / name).is_file()
                )
            for scope in ("validation", "held_out"):
                if scope == "held_out" and not fold.held_out_tasks:
                    continue
                report = paths.diagnostics(fold.name, seed, scope) / "alignment_pair.json"
                status = _field(report, "integrity", "status")
                if status != "pass":
                    fold_problems.append(f"paired diagnostics integrity is not 'pass': {report}")
                    continue
                comparisons = _field(report, "comparisons") or {}
                alignment_outcomes[f"{fold.name}/seed_{seed}/{scope}"] = {
                    name: value.get("evidence") for name, value in comparisons.items()
                }
                summary = paths.rollouts(scope, fold.name, seed) / "summary.json"
                if _field(paths.rollouts(scope, fold.name, seed) / "manifest.json", "finished") is not True:
                    fold_problems.append(f"generated-policy rollouts are not finished: {summary.parent}")
    add(
        "alignment_folds_complete_with_integrity",
        fold_problems,
        expected_folds=len(manifest.folds),
        # Recorded scientific outcomes; none of them is required to be favourable.
        recorded_outcomes=alignment_outcomes,
    )

    # ----------------------------------------------------- downstream gates
    one_task = tuple((study / "gates/one_task").rglob("one_task_gate.json"))
    add(
        "one_task_learning_gate",
        ["expected exactly one one-task gate with integrity passed"]
        if len(one_task) != 1 or _field(one_task[0], "integrity_passed") is not True
        else [],
        observed=len(one_task),
    )
    two_task = tuple((study / "gates/continual/two_task_gate").rglob("two_task_gate.json"))
    add(
        "two_task_shared_state_gate",
        ["expected exactly one passing two-task gate"]
        if len(two_task) != 1 or _field(two_task[0], "passed") is not True
        else [],
        observed=len(two_task),
    )
    fold_files = []
    for suite in REQUIRED_SUITES:
        fold_root = study / "folds" / suite
        fold_files.extend(
            f"missing {fold_root / name}"
            for name in ("ood_reference_bank.npz", "gamma_selection.json", "gamma_selection.report.json")
            if not (fold_root / name).is_file()
        )
    add("downstream_fold_artifacts", fold_files, expected_folds=len(REQUIRED_SUITES))

    # ------------------------------------------------------- downstream grids
    def audit_manifest_runs(name: str, leaves: list[tuple[Path, str, int]]) -> None:
        problems = []
        observed_records = 0
        for leaf, record_name, expected_records in leaves:
            manifest_path = leaf / "manifest.json"
            record_path = leaf / record_name
            if not manifest_path.is_file() or not record_path.is_file():
                problems.append(f"missing completed run artifacts: {leaf}")
                continue
            try:
                run_manifest = _json(manifest_path)
                if not run_manifest.get("finished_at") or run_manifest.get("failures"):
                    problems.append(f"unfinished or failed run: {leaf}")
                if run_manifest.get("git_dirty") is not False:
                    problems.append(f"run used a dirty Git checkout: {leaf}")
                commits.add(str(run_manifest.get("git_commit", "")))
                base_hashes.add(str(run_manifest.get("base_sha256", "")))
                count = _line_count(record_path)
                observed_records += count
                if count != expected_records:
                    problems.append(
                        f"wrong record count in {record_path}: expected {expected_records}, found {count}"
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"{leaf}: {exc}")
        add(
            name,
            problems,
            expected_runs=len(leaves),
            expected_records=sum(item[2] for item in leaves),
            observed_records=observed_records,
        )

    # Lower triangle of the stage-by-task matrix for a ten-task sequence.
    triangle = TASKS_PER_SUITE * (TASKS_PER_SUITE + 1) // 2
    audit_manifest_runs(
        "development_pilot_run_grid",
        [
            (
                study / "development/continual/ten_task_study" / PILOT_SUITE / f"seed_{seed}" / condition,
                "evaluation.jsonl",
                triangle,
            )
            for seed in seeds
            for condition in CONDITIONS
        ],
    )
    audit_manifest_runs(
        "continual_run_grid",
        [
            (
                publication / "continual/ten_task_study" / suite / f"seed_{seed}" / condition,
                "evaluation.jsonl",
                triangle,
            )
            for suite in REQUIRED_SUITES
            for seed in seeds
            for condition in CONDITIONS
        ],
    )
    audit_manifest_runs(
        "ood_run_grid",
        [
            (publication / "ood" / suite / f"seed_{seed}", "evaluation.jsonl", 5 * TASKS_PER_SUITE)
            for suite in REQUIRED_SUITES
            for seed in seeds
        ],
    )
    audit_manifest_runs(
        "component_swap_run_grid",
        [
            (
                publication / "component_swaps" / suite / f"seed_{seed}" / "proposed_asymmetric",
                "component_swaps.jsonl",
                3 * (TASKS_PER_SUITE - 1) * TASKS_PER_SUITE,
            )
            for suite in REQUIRED_SUITES
            for seed in seeds
        ],
    )
    recovery_problems = []
    for suite in REQUIRED_SUITES:
        for seed in seeds:
            path = (
                publication / "recovery" / suite / f"seed_{seed}"
                / "sequential_no_regularization" / "recovery_summary.json"
            )
            if not path.is_file():
                recovery_problems.append(f"missing {path}")
            elif _field(path, "transition_count") != TASKS_PER_SUITE - 1:
                recovery_problems.append(f"recovery transition_count is not 9: {path}")
    add("recovery_run_grid", recovery_problems, expected_runs=len(REQUIRED_SUITES) * len(seeds))

    report_files = (
        "continual_study.json",
        "continual_records.parquet",
        "ood_study.json",
        "ood_records.parquet",
        "component_swaps.json",
        "component_swaps.parquet",
        "component_drift_scatter.png",
    )
    report_root = publication / "reports"
    add(
        "publication_reports",
        [f"missing {report_root / name}" for name in report_files if not (report_root / name).is_file()],
        expected_files=len(report_files),
    )
    provenance_problems = []
    if len(commits) != 1 or "" in commits:
        provenance_problems.append(f"publication runs span {len(commits - {''})} Git commits")
    if len(base_hashes) != 1 or "" in base_hashes:
        provenance_problems.append(
            f"publication runs span {len(base_hashes - {''})} base checkpoint hashes"
        )
    add(
        "publication_provenance_consistency",
        provenance_problems,
        git_commits=sorted(commits - {""}),
        base_sha256=sorted(base_hashes - {""}),
    )
    return {
        "schema_version": 2,
        "experiment": manifest.name,
        "manifest_sha256": manifest.sha256(),
        "root": str(root.resolve()),
        "ready": all(check["passed"] for check in checks),
        "passed_checks": sum(check["passed"] for check in checks),
        "total_checks": len(checks),
        "checks": checks,
        "note": (
            "ready means complete and internally consistent. It does not mean that any "
            "method outperformed another; see recorded_outcomes and the study reports."
        ),
    }
