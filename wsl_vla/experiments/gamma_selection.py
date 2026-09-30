"""Validation-only selection of continual regularization hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .conditions import GammaSelection, MODALITIES, save_gamma_selection
from .protocol import REQUIRED_SUITES


@dataclass(frozen=True)
class GammaValidationRecord:
    run_id: str
    held_out_suite: str
    source_suite: str
    seed: int
    family: str
    gammas: Mapping[str, float]
    early_stopping_patience: int
    average_success_rate: float
    negative_backward_transfer: float
    validation_task_ids: Sequence[str]
    rollout_count: int
    stage_update_steps: Sequence[int]
    max_steps_per_stage: int
    alignment_checkpoint_sha256: str
    synthetic: bool = False
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1 or self.synthetic:
            raise ValueError("gamma selection accepts only schema-1 real validation records")
        if self.held_out_suite not in REQUIRED_SUITES:
            raise ValueError("unknown held-out suite")
        if self.source_suite not in set(REQUIRED_SUITES) - {self.held_out_suite}:
            raise ValueError("gamma validation source must be a non-held-out suite")
        if self.family not in {"proposed", "uniform"}:
            raise ValueError("gamma family must be proposed or uniform")
        if set(self.gammas) != set(MODALITIES):
            raise ValueError("gamma record must contain every modality")
        values = {name: float(self.gammas[name]) for name in MODALITIES}
        if min(values.values()) < 0:
            raise ValueError("gamma values cannot be negative")
        if self.family == "uniform" and len(set(values.values())) != 1:
            raise ValueError("uniform gamma candidates must be equal")
        if self.family == "proposed" and not (
            values["vision"] > values["action"]
            and values["language"] > values["action"]
        ):
            raise ValueError("proposed gamma candidates must preserve the hypothesized direction")
        if self.early_stopping_patience <= 0 or self.rollout_count <= 0:
            raise ValueError("patience and rollout count must be positive")
        if len(self.validation_task_ids) != 2 or len(set(self.validation_task_ids)) != 2:
            raise ValueError("each source suite must use exactly two locked validation tasks")
        if tuple(self.validation_task_ids) != (
            f"{self.source_suite}_8",
            f"{self.source_suite}_9",
        ):
            raise ValueError("gamma records must use locked source task indices 8 and 9")
        if (
            self.max_steps_per_stage <= 0
            or len(self.stage_update_steps) != len(self.validation_task_ids)
            or any(
                int(value) <= 0 or int(value) > self.max_steps_per_stage
                for value in self.stage_update_steps
            )
        ):
            raise ValueError("gamma records require valid early-stopped step counts")
        if not 0 <= self.average_success_rate <= 1:
            raise ValueError("validation success rate must lie in [0, 1]")
        if not np.isfinite(self.negative_backward_transfer):
            raise ValueError("validation NBT must be finite")
        if len(self.alignment_checkpoint_sha256) != 64:
            raise ValueError("alignment checkpoint identity must be SHA-256")

    @property
    def selection_score(self) -> float:
        self.validate()
        return float(self.average_success_rate - self.negative_backward_transfer)


def append_gamma_validation_record(
    record: GammaValidationRecord, path: str | Path
) -> None:
    record.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load_gamma_validation_records(path: str | Path) -> tuple[GammaValidationRecord, ...]:
    records = []
    source = Path(path)
    files = tuple(sorted(source.rglob("gamma_validation.jsonl"))) if source.is_dir() else (source,)
    if not files:
        raise FileNotFoundError(f"no gamma_validation.jsonl records below {source}")
    for file in files:
        with file.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = GammaValidationRecord(**json.loads(line))
                    record.validate()
                except Exception as exc:
                    raise ValueError(
                        f"invalid gamma validation record in {file} line {line_number}"
                    ) from exc
                records.append(record)
    if not records:
        raise ValueError("gamma validation record file is empty")
    return tuple(records)


def _candidate_key(record: GammaValidationRecord):
    return (
        record.family,
        tuple(float(record.gammas[name]) for name in MODALITIES),
        int(record.early_stopping_patience),
    )


def select_gamma_configuration(
    records: Iterable[GammaValidationRecord],
    *,
    held_out_suite: str,
    seeds: Sequence[int],
) -> tuple[GammaSelection, dict]:
    """Select only after every candidate has all suite-by-seed validation cells."""

    values = tuple(records)
    requested_seeds = tuple(int(value) for value in seeds)
    if len(requested_seeds) != 3 or len(set(requested_seeds)) != 3:
        raise ValueError("gamma selection requires exactly three independent seeds")
    source_suites = tuple(sorted(set(REQUIRED_SUITES) - {held_out_suite}))
    if any(record.held_out_suite != held_out_suite for record in values):
        raise ValueError("gamma records combine different held-out folds")
    identities = set()
    grouped: dict[tuple, list[GammaValidationRecord]] = {}
    alignment_hashes = set()
    rollout_counts = set()
    maximum_steps = set()
    for record in values:
        record.validate()
        if record.source_suite not in source_suites or record.seed not in requested_seeds:
            raise ValueError("gamma records contain an unexpected suite or seed")
        identity = (_candidate_key(record), record.source_suite, record.seed)
        if identity in identities:
            raise ValueError("duplicate gamma validation candidate/suite/seed cell")
        identities.add(identity)
        grouped.setdefault(_candidate_key(record), []).append(record)
        alignment_hashes.add(record.alignment_checkpoint_sha256)
        rollout_counts.add(record.rollout_count)
        maximum_steps.add(record.max_steps_per_stage)
    if len(alignment_hashes) != 1:
        raise ValueError("gamma validation records use different alignment checkpoints")
    if len(rollout_counts) != 1 or len(maximum_steps) != 1:
        raise ValueError("gamma candidates use inconsistent rollout or maximum-step budgets")
    expected_cells = {
        (suite, seed) for suite in source_suites for seed in requested_seeds
    }
    summaries = []
    families_by_patience: dict[int, set[str]] = {}
    for key, candidate_records in grouped.items():
        cells = {(record.source_suite, record.seed) for record in candidate_records}
        if cells != expected_cells:
            raise ValueError(
                f"incomplete gamma candidate {key}: expected {len(expected_cells)} cells, "
                f"found {len(cells)}"
            )
        family, gamma_tuple, patience = key
        families_by_patience.setdefault(patience, set()).add(family)
        scores = np.asarray([record.selection_score for record in candidate_records])
        summaries.append(
            {
                "family": family,
                "gammas": dict(zip(MODALITIES, gamma_tuple)),
                "early_stopping_patience": patience,
                "mean_selection_score": float(scores.mean()),
                "score_standard_error": float(scores.std(ddof=1) / np.sqrt(scores.size)),
                "cell_count": int(scores.size),
            }
        )
    for patience, families in families_by_patience.items():
        if families != {"proposed", "uniform"}:
            raise ValueError(f"patience {patience} lacks proposed or uniform candidates")

    patience_results = []
    for patience in sorted(families_by_patience):
        proposed = [
            item for item in summaries
            if item["early_stopping_patience"] == patience and item["family"] == "proposed"
        ]
        uniform = [
            item for item in summaries
            if item["early_stopping_patience"] == patience and item["family"] == "uniform"
        ]
        best_proposed = max(
            proposed,
            key=lambda item: (item["mean_selection_score"], -sum(item["gammas"].values())),
        )
        best_uniform = max(
            uniform,
            key=lambda item: (item["mean_selection_score"], -sum(item["gammas"].values())),
        )
        patience_results.append(
            {
                "patience": patience,
                "joint_score": (
                    best_proposed["mean_selection_score"]
                    + best_uniform["mean_selection_score"]
                ) / 2,
                "proposed": best_proposed,
                "uniform": best_uniform,
            }
        )
    selected = max(
        patience_results,
        key=lambda item: (item["joint_score"], -item["patience"]),
    )
    selection = GammaSelection(
        held_out_suite=held_out_suite,
        validation_suites=source_suites,
        proposed=selected["proposed"]["gammas"],
        uniform=float(selected["uniform"]["gammas"]["vision"]),
        early_stopping_patience=int(selected["patience"]),
    )
    selection.validate()
    report = {
        "schema_version": 1,
        "held_out_suite": held_out_suite,
        "seeds": list(requested_seeds),
        "source_suites": list(source_suites),
        "alignment_checkpoint_sha256": next(iter(alignment_hashes)),
        "rollout_count": next(iter(rollout_counts)),
        "max_steps_per_stage": next(iter(maximum_steps)),
        "selection_objective": "mean_success_rate_minus_negative_backward_transfer",
        "candidate_summaries": summaries,
        "patience_summaries": patience_results,
        "selected": {
            "proposed": dict(selection.proposed),
            "uniform": selection.uniform,
            "early_stopping_patience": selection.early_stopping_patience,
        },
    }
    return selection, report


def save_gamma_selection_with_report(
    selection: GammaSelection,
    report: Mapping,
    *,
    selection_path: str | Path,
    report_path: str | Path,
) -> None:
    save_gamma_selection(selection, selection_path)
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
