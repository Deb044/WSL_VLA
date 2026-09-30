"""Deterministic distributed job graph for the full Methodology 1 study."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .protocol import REQUIRED_SUITES
from .study_audit import CONDITIONS, SEEDS


def build_methodology1_job_plan(
    *,
    research_root: str | Path,
    data_files: Mapping[str, Sequence[str | Path]],
    gamma_candidates: Mapping[str, object],
    python: str = "python",
) -> tuple[dict, ...]:
    root = Path(research_root).resolve()
    publication = root / "publication"
    jobs: list[dict] = []

    def add(job_id, stage, command, dependencies=(), outputs=(), gpu=True):
        jobs.append(
            {
                "schema_version": 1,
                "job_id": job_id,
                "stage": stage,
                "command": [str(value) for value in command],
                "dependencies": list(dependencies),
                "expected_outputs": [str(Path(value).resolve()) for value in outputs],
                "resources": {"gpu_required": gpu},
            }
        )

    add(
        "preflight",
        "gates",
        [python, "scripts/research_preflight.py", "--output-dir", root / "preflight"],
        outputs=[root / "preflight/preflight.json"],
        gpu=False,
    )
    add(
        "zero_adapter_equivalence",
        "gates",
        [python, "scripts/verify_official_octo.py", "--output", root / "preflight/octo_equivalence.json"],
        dependencies=["preflight"],
        outputs=[root / "preflight/octo_equivalence.json"],
    )
    add(
        "one_task_gate",
        "gates",
        [python, "scripts/run_one_task_learning_gate.py", "--suite", "libero_spatial", "--task-index", 0, "--seed", 17, "--output-root", root / "gates/one_task"],
        dependencies=["zero_adapter_equivalence"],
    )
    add(
        "two_task_gate",
        "gates",
        [python, "scripts/run_official_continual.py", "--suite", "libero_spatial", "--condition", "sequential_no_regularization", "--seed", 17, "--task-count", 2, "--output-root", root / "gates/continual"],
        dependencies=["one_task_gate"],
    )

    evidence_ids = []
    zoo_ids = []
    for suite in REQUIRED_SUITES:
        files = tuple(data_files.get(suite, ()))
        if len(files) != 10:
            raise ValueError(f"{suite} must resolve exactly ten ordered data files")
        for task_index, data_file in enumerate(files):
            evidence_id = f"evidence:{suite}:{task_index}"
            evidence_path = root / "evidence" / suite / f"{suite}_{task_index}.npz"
            add(
                evidence_id,
                "evidence",
                [python, "scripts/extract_research_evidence.py", "--data-file", data_file, "--suite", suite, "--task-index", task_index, "--output", evidence_path],
                dependencies=["two_task_gate"],
                outputs=[evidence_path],
            )
            evidence_ids.append(evidence_id)
            for seed in SEEDS:
                job_id = f"zoo:{suite}:{task_index}:seed-{seed}"
                add(
                    job_id,
                    "model_zoo",
                    [python, "scripts/train_research_zoo.py", "--suite", suite, "--task-index", task_index, "--seed", seed, "--evidence", evidence_path, "--output-root", root / "population"],
                    dependencies=[evidence_id],
                )
                zoo_ids.append(job_id)
    add(
        "verify_research_zoo",
        "model_zoo",
        [python, "scripts/verify_research_zoo.py", root / "population", "--output", root / "population/verification.json"],
        dependencies=zoo_ids,
        outputs=[root / "population/verification.json"],
        gpu=False,
    )

    fold_gate_ids = {}
    bank_ids = {}
    selection_ids = {}
    all_gamma_ids = []
    patience_values = tuple(int(value) for value in gamma_candidates["early_stopping_patience"])
    for held_out in REQUIRED_SUITES:
        fold = root / "alignment" / held_out
        archive_id = f"archive:{held_out}"
        add(
            archive_id,
            "alignment",
            [python, "scripts/build_alignment_archive.py", root / "population", "--held-out-suite", held_out, "--validation-task-indices", "8,9", "--output", fold / "archive.npz"],
            dependencies=["verify_research_zoo"],
            outputs=[fold / "archive.npz", fold / "archive.manifest.json"],
            gpu=False,
        )
        variant_ids = []
        for variant in ("aligned", "reconstruction_only"):
            job_id = f"alignment:{held_out}:{variant}"
            command = [python, "scripts/train_research_alignment.py", fold / "archive.npz", "--output", fold / variant]
            if variant == "reconstruction_only":
                command += ["--contrastive-weight", 0]
            add(job_id, "alignment", command, dependencies=[archive_id], outputs=[fold / variant / "metadata.json"])
            variant_ids.append(job_id)
        gate_id = f"alignment_gate:{held_out}"
        add(
            gate_id,
            "alignment",
            [python, "scripts/run_alignment_advantage_gate.py", fold / "archive.npz", "--aligned-checkpoint", fold / "aligned", "--reconstruction-checkpoint", fold / "reconstruction_only", "--output", fold / "alignment_advantage.json"],
            dependencies=variant_ids,
            outputs=[fold / "alignment_advantage.json"],
        )
        fold_gate_ids[held_out] = gate_id
        bank_id = f"ood_bank:{held_out}"
        add(
            bank_id,
            "alignment",
            [python, "scripts/build_ood_reference_bank.py", root / "population", fold / "aligned", "--held-out-suite", held_out, "--output", fold / "ood_reference_bank.npz"],
            dependencies=[gate_id],
            outputs=[fold / "ood_reference_bank.npz"],
            gpu=False,
        )
        bank_ids[held_out] = bank_id

        gamma_ids = []
        for family in ("proposed", "uniform"):
            for candidate in gamma_candidates[family]:
                for patience in patience_values:
                    for source in (suite for suite in REQUIRED_SUITES if suite != held_out):
                        for seed in SEEDS:
                            job_id = f"gamma:{held_out}:{source}:seed-{seed}:{family}:{candidate['name']}:p-{patience}"
                            add(
                                job_id,
                                "gamma_validation",
                                [python, "scripts/run_gamma_validation.py", "--held-out-suite", held_out, "--source-suite", source, "--seed", seed, "--family", family, "--gamma-vision", candidate["vision"], "--gamma-language", candidate["language"], "--gamma-action", candidate["action"], "--patience", patience, "--alignment-checkpoint", fold / "aligned", "--evidence-root", root / "evidence", "--output-root", root / "gamma_validation"],
                                dependencies=[gate_id],
                            )
                            gamma_ids.append(job_id)
                            all_gamma_ids.append(job_id)
        selection_id = f"gamma_select:{held_out}"
        add(
            selection_id,
            "gamma_selection",
            [python, "scripts/select_validation_gammas.py", root / "gamma_validation" / held_out, "--held-out-suite", held_out, "--seeds", "17,42,73", "--output", fold / "gamma_selection.json", "--report", fold / "gamma_selection.report.json"],
            dependencies=gamma_ids,
            outputs=[fold / "gamma_selection.json", fold / "gamma_selection.report.json"],
            gpu=False,
        )
        selection_ids[held_out] = selection_id

    pilot_ids = []
    pilot_suite = "libero_spatial"
    pilot_fold = root / "alignment" / pilot_suite
    for seed in SEEDS:
        for condition in CONDITIONS:
            job_id = f"pilot:{pilot_suite}:seed-{seed}:{condition}"
            add(
                job_id,
                "pilot",
                [python, "scripts/run_official_continual.py", "--suite", pilot_suite, "--seed", seed, "--condition", condition, "--gamma-selection", pilot_fold / "gamma_selection.json", "--alignment-checkpoint", pilot_fold / "aligned", "--reconstruction-checkpoint", pilot_fold / "reconstruction_only", "--evidence-root", root / "evidence", "--tier", "development", "--output-root", root / "development/continual"],
                dependencies=[selection_ids[pilot_suite]],
            )
            pilot_ids.append(job_id)

    continual_ids = []
    ood_ids = []
    swap_ids = []
    recovery_ids = []
    for suite in REQUIRED_SUITES:
        fold = root / "alignment" / suite
        for seed in SEEDS:
            for condition in CONDITIONS:
                job_id = f"continual:{suite}:seed-{seed}:{condition}"
                add(
                    job_id,
                    "continual",
                    [python, "scripts/run_official_continual.py", "--suite", suite, "--seed", seed, "--condition", condition, "--gamma-selection", fold / "gamma_selection.json", "--alignment-checkpoint", fold / "aligned", "--reconstruction-checkpoint", fold / "reconstruction_only", "--evidence-root", root / "evidence", "--tier", "publication", "--output-root", publication / "continual"],
                    dependencies=[selection_ids[suite], *pilot_ids],
                )
                continual_ids.append(job_id)
            ood_id = f"ood:{suite}:seed-{seed}"
            add(
                ood_id,
                "ood",
                [python, "scripts/run_official_ood.py", "--suite", suite, "--seed", seed, "--alignment-checkpoint", fold / "aligned", "--reference-bank", fold / "ood_reference_bank.npz", "--evidence-root", root / "evidence", "--tier", "publication", "--output-root", publication / "ood"],
                dependencies=[bank_ids[suite]],
            )
            ood_ids.append(ood_id)
            source = publication / "continual/ten_task_study" / suite / f"seed_{seed}"
            swap_id = f"component_swap:{suite}:seed-{seed}"
            add(
                swap_id,
                "mechanistic_probes",
                [python, "scripts/run_component_swaps.py", source / "proposed_asymmetric", "--tier", "publication", "--output-root", publication / "component_swaps"],
                dependencies=[f"continual:{suite}:seed-{seed}:proposed_asymmetric"],
            )
            swap_ids.append(swap_id)
            recovery_id = f"recovery:{suite}:seed-{seed}"
            add(
                recovery_id,
                "mechanistic_probes",
                [python, "scripts/run_recovery_probe.py", source / "sequential_no_regularization", "--output-root", publication / "recovery"],
                dependencies=[f"continual:{suite}:seed-{seed}:sequential_no_regularization"],
            )
            recovery_ids.append(recovery_id)

    report_root = publication / "reports"
    add("report:continual", "reporting", [python, "scripts/report_publication_study.py", publication / "continual/ten_task_study", "--output-json", report_root / "continual_study.json", "--output-parquet", report_root / "continual_records.parquet"], dependencies=continual_ids, gpu=False)
    add("report:ood", "reporting", [python, "scripts/report_ood_study.py", publication / "ood", "--output-json", report_root / "ood_study.json", "--output-parquet", report_root / "ood_records.parquet"], dependencies=ood_ids, gpu=False)
    add("report:components", "reporting", [python, "scripts/report_component_swaps.py", publication / "component_swaps", "--output-json", report_root / "component_swaps.json", "--output-parquet", report_root / "component_swaps.parquet", "--output-figure", report_root / "component_drift_scatter.png"], dependencies=swap_ids, gpu=False)
    add("audit", "reporting", [python, "scripts/audit_methodology1_study.py", root, "--output", report_root / "completion_audit.json"], dependencies=["report:continual", "report:ood", "report:components", *recovery_ids], outputs=[report_root / "completion_audit.json"], gpu=False)

    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise AssertionError("generated duplicate job IDs")
    known = set(ids)
    if any(set(job["dependencies"]) - known for job in jobs):
        raise AssertionError("generated dependency on an unknown job")
    return tuple(jobs)


def summarize_job_plan(jobs: Sequence[Mapping[str, object]]) -> dict:
    return {
        "schema_version": 1,
        "job_count": len(jobs),
        "gpu_job_count": sum(bool(job["resources"]["gpu_required"]) for job in jobs),
        "jobs_by_stage": dict(sorted(Counter(str(job["stage"]) for job in jobs).items())),
    }
