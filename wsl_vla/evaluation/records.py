"""Machine-readable evaluation record persistence and matrix construction."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..contracts import EvaluationRecord


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
