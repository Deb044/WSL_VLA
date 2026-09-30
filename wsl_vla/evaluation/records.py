"""Machine-readable evaluation record persistence and matrix construction."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..contracts import ComponentSwapRecord, EvaluationRecord, OODEvaluationRecord


def append_evaluation_record(record: EvaluationRecord, path: str | Path) -> None:
    _ = record.success_rate
    if len(record.checkpoint_sha256) != 64:
        raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load_evaluation_records(path: str | Path) -> tuple[EvaluationRecord, ...]:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(EvaluationRecord(**json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid evaluation record on line {line_number}") from exc
    return tuple(records)


def load_evaluation_record_tree(path: str | Path) -> tuple[EvaluationRecord, ...]:
    """Load one JSONL file or recursively collect canonical run record files."""

    source = Path(path)
    if source.is_file():
        return load_evaluation_records(source)
    if not source.is_dir():
        raise FileNotFoundError(f"evaluation record path is missing: {source}")
    files = tuple(sorted(source.rglob("evaluation.jsonl")))
    if not files:
        raise FileNotFoundError(f"no evaluation.jsonl files below {source}")
    return tuple(record for file in files for record in load_evaluation_records(file))


def append_ood_evaluation_record(record: OODEvaluationRecord, path: str | Path) -> None:
    _append_validated_record(record, path, validation=lambda item: item.success_rate)


def load_ood_evaluation_records(path: str | Path) -> tuple[OODEvaluationRecord, ...]:
    return _load_typed_records(path, OODEvaluationRecord)


def load_ood_evaluation_record_tree(path: str | Path) -> tuple[OODEvaluationRecord, ...]:
    return _load_typed_record_tree(path, "evaluation.jsonl", OODEvaluationRecord)


def append_component_swap_record(record: ComponentSwapRecord, path: str | Path) -> None:
    _append_validated_record(record, path, validation=lambda item: item.success_rate_drop)


def load_component_swap_records(path: str | Path) -> tuple[ComponentSwapRecord, ...]:
    return _load_typed_records(path, ComponentSwapRecord)


def load_component_swap_record_tree(path: str | Path) -> tuple[ComponentSwapRecord, ...]:
    return _load_typed_record_tree(path, "component_swaps.jsonl", ComponentSwapRecord)


def _append_validated_record(record, path: str | Path, *, validation) -> None:
    validation(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _load_typed_records(path: str | Path, record_type):
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = record_type(**json.loads(line))
                if isinstance(record, OODEvaluationRecord):
                    _ = record.success_rate
                else:
                    _ = record.success_rate_drop
                records.append(record)
            except Exception as exc:
                raise ValueError(f"invalid {record_type.__name__} on line {line_number}") from exc
    return tuple(records)


def _load_typed_record_tree(path: str | Path, filename: str, record_type):
    source = Path(path)
    if source.is_file():
        return _load_typed_records(source, record_type)
    if not source.is_dir():
        raise FileNotFoundError(f"record path is missing: {source}")
    files = tuple(sorted(source.rglob(filename)))
    if not files:
        raise FileNotFoundError(f"no {filename} files below {source}")
    return tuple(record for file in files for record in _load_typed_records(file, record_type))


def export_typed_records_parquet(records: Iterable, path: str | Path) -> None:
    """Export OOD or component-swap records with derived rates."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the research extra to export Parquet") from exc
    rows = []
    for record in records:
        row = asdict(record)
        if isinstance(record, OODEvaluationRecord):
            row["success_rate"] = record.success_rate
        elif isinstance(record, ComponentSwapRecord):
            row["success_rate_drop"] = record.success_rate_drop
        else:
            raise TypeError("typed Parquet export accepts OOD or component-swap records")
        for name, value in row.pop("latent_drift", {}).items():
            row[f"latent_drift_{name}"] = value
        rows.append(row)
    if not rows:
        raise ValueError("cannot export an empty record set")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)


def records_to_success_matrix(
    records: Iterable[EvaluationRecord],
    *,
    suite: str,
    seed: int,
    condition: str,
    task_count: int = 10,
) -> np.ndarray:
    selected = [
        record
        for record in records
        if record.suite == suite and record.seed == seed and record.condition == condition
    ]
    matrix = np.full((task_count, task_count), np.nan, dtype=np.float64)
    seen = set()
    for record in selected:
        key = (record.training_stage, record.evaluated_task_index)
        if key in seen:
            raise ValueError(f"duplicate evaluation cell {key}")
        seen.add(key)
        if not 0 <= record.evaluated_task_index <= record.training_stage < task_count:
            raise ValueError(f"record lies outside lower-triangular protocol: {key}")
        matrix[key] = record.success_rate
    expected = {(stage, task) for stage in range(task_count) for task in range(stage + 1)}
    missing = expected - seen
    if missing:
        raise ValueError(f"incomplete success matrix; missing {len(missing)} cells")
    return matrix


def export_evaluation_parquet(
    records: Iterable[EvaluationRecord], path: str | Path
) -> None:
    """Persist normalized rollout records; requires the research parquet extra."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the research extra to export Parquet") from exc
    rows = []
    for record in records:
        row = asdict(record)
        row["success_rate"] = record.success_rate
        for name, value in row.pop("latent_drift").items():
            row[f"latent_drift_{name}"] = value
        rows.append(row)
    if not rows:
        raise ValueError("cannot export an empty evaluation record set")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
