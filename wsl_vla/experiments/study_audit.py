"""Machine-checkable completion audit for the Methodology 1 study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .protocol import REQUIRED_SUITES


SEEDS = (17, 42, 73)
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


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _json_field_equals(path: Path, field: str, expected: Any) -> bool:
    try:
        return _json(path).get(field) == expected
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def audit_methodology1_study(root: str | Path) -> dict[str, Any]:
    """Audit expected artifacts without importing JAX, Octo, or LIBERO."""
    root = Path(root)
    publication = root / "publication"
    checks: list[dict[str, Any]] = []
    commits: set[str] = set()
    base_hashes: set[str] = set()

    def add(name: str, problems: Iterable[str], **details: Any) -> None:
        items = list(problems)
        checks.append({"name": name, "passed": not items, "problems": items[:20], **details})

    for name, relative, success_key in (
        ("preflight", "preflight/preflight.json", "ok"),
        ("zero_adapter_equivalence", "preflight/octo_equivalence.json", "ok"),
        ("population_verification", "population/verification.json", "ok"),
    ):
        path = root / relative
        problems = []
        if not path.is_file():
            problems.append(f"missing {path}")
        else:
            try:
                payload = _json(path)
                if payload.get(success_key) is not True:
                    problems.append(f"{success_key} is not true")
                if name == "population_verification" and payload.get("sample_count") != 360:
                    problems.append("population sample_count is not 360")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                problems.append(str(exc))
        add(name, problems, path=str(path))

    one_task = tuple((root / "gates/one_task").rglob("one_task_gate.json"))
    add(
        "one_task_learning_gate",
        ["expected exactly one passing one-task gate"]
        if len(one_task) != 1 or not _json_field_equals(one_task[0], "passed", True)
        else [],
        observed=len(one_task),
    )
    two_task = tuple((root / "gates/continual/two_task_gate").rglob("two_task_gate.json"))
    add(
        "two_task_shared_state_gate",
        ["expected exactly one passing two-task gate"]
        if len(two_task) != 1 or not _json_field_equals(two_task[0], "passed", True)
        else [],
        observed=len(two_task),
    )

    fold_problems = []
    checkpoint_files = (
        "alignment_params.msgpack",
        "linear_mappers.npz",
        "empirical_shells.npz",
        "token_layout.npz",
        "metadata.json",
    )
    for suite in REQUIRED_SUITES:
        fold = root / "alignment" / suite
        required = [
            fold / "archive.npz",
            fold / "archive.manifest.json",
            fold / "alignment_advantage.json",
            fold / "ood_reference_bank.npz",
            fold / "gamma_selection.json",
            fold / "gamma_selection.report.json",
        ] + [fold / variant / name for variant in ("aligned", "reconstruction_only") for name in checkpoint_files]
        fold_problems.extend(f"missing {path}" for path in required if not path.is_file())
        gate = fold / "alignment_advantage.json"
        if gate.is_file() and not _json_field_equals(gate, "passed", True):
            fold_problems.append(f"alignment gate did not pass: {gate}")
    add("four_alignment_folds", fold_problems, expected=4)

    def audit_manifest_runs(
        name: str,
        leaves: list[tuple[Path, str, int]],
    ) -> None:
        problems = []
        observed_records = 0
        for leaf, record_name, expected_records in leaves:
            manifest_path = leaf / "manifest.json"
            record_path = leaf / record_name
            if not manifest_path.is_file() or not record_path.is_file():
                problems.append(f"missing completed run artifacts: {leaf}")
                continue
            try:
                manifest = _json(manifest_path)
                if not manifest.get("finished_at") or manifest.get("failures"):
                    problems.append(f"unfinished or failed run: {leaf}")
                if manifest.get("git_dirty") is not False:
                    problems.append(f"run used a dirty Git checkout: {leaf}")
                commits.add(str(manifest.get("git_commit", "")))
                base_hashes.add(str(manifest.get("base_sha256", "")))
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

    pilot = [
        (
            root / "development/continual/ten_task_study/libero_spatial" / f"seed_{seed}" / condition,
            "evaluation.jsonl",
            55,
        )
        for seed in SEEDS
        for condition in CONDITIONS
    ]
    audit_manifest_runs("development_pilot_24_run_grid", pilot)
    continual = [
        (
            publication / "continual/ten_task_study" / suite / f"seed_{seed}" / condition,
            "evaluation.jsonl",
            55,
        )
        for suite in REQUIRED_SUITES
        for seed in SEEDS
        for condition in CONDITIONS
    ]
    audit_manifest_runs("continual_96_run_grid", continual)
    ood = [
        (publication / "ood" / suite / f"seed_{seed}", "evaluation.jsonl", 50)
        for suite in REQUIRED_SUITES
        for seed in SEEDS
    ]
    audit_manifest_runs("ood_12_run_grid", ood)
    swaps = [
        (
            publication / "component_swaps" / suite / f"seed_{seed}" / "proposed_asymmetric",
            "component_swaps.jsonl",
            270,
        )
        for suite in REQUIRED_SUITES
        for seed in SEEDS
    ]
    audit_manifest_runs("component_swap_12_run_grid", swaps)

    recovery_problems = []
    for suite in REQUIRED_SUITES:
        for seed in SEEDS:
            path = (
                publication
                / "recovery"
                / suite
                / f"seed_{seed}"
                / "sequential_no_regularization"
                / "recovery_summary.json"
            )
            if not path.is_file():
                recovery_problems.append(f"missing {path}")
            elif not _json_field_equals(path, "transition_count", 9):
                recovery_problems.append(f"recovery transition_count is not 9: {path}")
    add("recovery_12_run_grid", recovery_problems, expected_runs=12)

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
        "schema_version": 1,
        "root": str(root.resolve()),
        "ready": all(check["passed"] for check in checks),
        "passed_checks": sum(check["passed"] for check in checks),
        "total_checks": len(checks),
        "checks": checks,
    }
