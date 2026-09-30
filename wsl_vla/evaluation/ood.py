"""Leakage-safe OOD adapter baselines and rollout protocol."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

import numpy as np

from ..contracts import OODEvaluationRecord
from .sequential import RolloutOutcome


OOD_METHODS = (
    "mapped_zero_shot",
    "mapped_latent_refinement",
    "mapped_weight_finetune",
    "nearest_neighbor",
    "mean_latent",
)
MODALITIES = ("vision", "language", "action")
State = TypeVar("State")


@dataclass(frozen=True)
class ReferenceAdapter:
    task_id: str
    suite: str
    evidence: Mapping[str, np.ndarray]
    latents: Mapping[str, np.ndarray]


class OODReferenceBank:
    """Training-suite-only baselines for nearest and mean latent retrieval."""

    def __init__(self, samples: Sequence[ReferenceAdapter], *, held_out_suite: str):
        if not samples:
            raise ValueError("OOD reference bank cannot be empty")
        if any(item.suite == held_out_suite for item in samples):
            raise ValueError("held-out suite adapters cannot enter the OOD reference bank")
        if len({item.task_id for item in samples}) != len(samples):
            raise ValueError("OOD reference bank task identities must be unique")
        for item in samples:
            if set(item.evidence) != set(MODALITIES) or set(item.latents) != set(MODALITIES):
                raise ValueError("reference adapters must contain every modality")
        self.samples = tuple(samples)
        self.held_out_suite = held_out_suite
        self.training_suites = tuple(sorted({item.suite for item in samples}))
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        for modality in MODALITIES:
            evidence_shapes = {np.asarray(item.evidence[modality]).shape for item in self.samples}
            latent_shapes = {np.asarray(item.latents[modality]).shape for item in self.samples}
            if len(evidence_shapes) != 1 or len(latent_shapes) != 1:
                raise ValueError(f"inconsistent {modality} shapes in OOD reference bank")

    def nearest_neighbor_latents(
        self, query_evidence: Mapping[str, np.ndarray]
    ) -> tuple[str, dict[str, np.ndarray]]:
        if set(query_evidence) != set(MODALITIES):
            raise ValueError("query evidence must contain every modality")
        distances = []
        for sample in self.samples:
            distance = 0.0
            for modality in MODALITIES:
                query = np.asarray(query_evidence[modality], dtype=np.float64)
                reference = np.asarray(sample.evidence[modality], dtype=np.float64)
                if query.shape != reference.shape:
                    raise ValueError(f"query {modality} evidence shape differs from bank")
                scale = max(float(np.linalg.norm(reference)), 1e-8)
                distance += float(np.linalg.norm(query - reference) / scale)
            distances.append(distance)
        index = int(np.argmin(distances))
        selected = self.samples[index]
        return selected.task_id, {
            name: np.array(selected.latents[name], copy=True) for name in MODALITIES
        }

    def mean_latents(self) -> dict[str, np.ndarray]:
        return {
            name: np.mean(
                np.stack([np.asarray(item.latents[name]) for item in self.samples]), axis=0
            )
            for name in MODALITIES
        }


@dataclass(frozen=True)
class OODCandidate(Generic[State]):
    state: State
    checkpoint_sha256: str
    adaptation_steps: int
    adaptation_loss: float | None = None
    adapter_bytes: int = 0
    evidence_bytes: int = 0


CandidateBuilder = Callable[[str, str, int], OODCandidate[State]]
Rollout = Callable[[State, str, int, int], RolloutOutcome]
RecordSink = Callable[[OODEvaluationRecord], None]


def aggregate_reference_adapters(
    *,
    task_ids: Sequence[str],
    suites: Sequence[str],
    evidence: Mapping[str, np.ndarray],
    latents: Mapping[str, np.ndarray],
    held_out_suite: str,
) -> OODReferenceBank:
    """Average repeated zoo checkpoints into one reference per source task."""

    sample_count = len(task_ids)
    if sample_count == 0 or len(suites) != sample_count:
        raise ValueError("reference identities must be non-empty and aligned")
    if set(evidence) != set(MODALITIES) or set(latents) != set(MODALITIES):
        raise ValueError("reference arrays must contain every modality")
    if any(np.asarray(values).shape[0] != sample_count for values in (*evidence.values(), *latents.values())):
        raise ValueError("reference arrays must share the sample dimension")
    grouped: dict[str, list[int]] = {}
    task_suite: dict[str, str] = {}
    for index, (task_id, suite) in enumerate(zip(task_ids, suites)):
        if suite == held_out_suite:
            raise ValueError("held-out suite adapters cannot enter the OOD reference bank")
        if task_id in task_suite and task_suite[task_id] != suite:
            raise ValueError("one reference task identity appears in multiple suites")
        task_suite[task_id] = suite
        grouped.setdefault(task_id, []).append(index)
    samples = []
    for task_id in sorted(grouped):
        indices = np.asarray(grouped[task_id], dtype=np.int64)
        samples.append(
            ReferenceAdapter(
                task_id=task_id,
                suite=task_suite[task_id],
                evidence={
                    name: np.asarray(evidence[name])[indices].mean(axis=0)
                    for name in MODALITIES
                },
                latents={
                    name: np.asarray(latents[name])[indices].mean(axis=0)
                    for name in MODALITIES
                },
            )
        )
    return OODReferenceBank(samples, held_out_suite=held_out_suite)


def run_ood_protocol(
    *,
    task_ids: Sequence[str],
    instructions: Sequence[str],
    candidate_builders: Mapping[str, CandidateBuilder[State]],
    rollout: Rollout[State],
    run_id: str,
    held_out_suite: str,
    seed: int,
    rollout_count: int,
    matched_adaptation_steps: int,
    source_training_suites: Sequence[str],
    alignment_checkpoint_sha256: str,
    record_sink: RecordSink | None = None,
) -> tuple[OODEvaluationRecord, ...]:
    """Evaluate all five OOD methods on identical fixed initializations."""

    if tuple(candidate_builders) != OOD_METHODS:
        raise ValueError(f"OOD builders must be ordered exactly as {OOD_METHODS}")
    if len(task_ids) != len(instructions) or not task_ids:
        raise ValueError("OOD task identities and instructions must align")
    if len(set(task_ids)) != len(task_ids) or rollout_count <= 0:
        raise ValueError("OOD tasks must be unique and rollouts positive")
    if matched_adaptation_steps <= 0:
        raise ValueError("matched adaptation steps must be positive")
    if held_out_suite in source_training_suites:
        raise ValueError("held-out suite cannot be an OOD training source")
    if len(source_training_suites) != 3 or len(set(source_training_suites)) != 3:
        raise ValueError("OOD evaluation requires exactly three distinct source suites")
    records = []
    for task_id, instruction in zip(task_ids, instructions):
        reference_initializations = None
        for method in OOD_METHODS:
            started = perf_counter()
            candidate = candidate_builders[method](task_id, instruction, seed)
            if method in {"mapped_latent_refinement", "mapped_weight_finetune"}:
                if candidate.adaptation_steps != matched_adaptation_steps:
                    raise ValueError("latent and weight refinement must use matched steps")
            elif candidate.adaptation_steps != 0:
                raise ValueError(f"{method} must not consume adaptation steps")
            outcome = rollout(candidate.state, task_id, rollout_count, seed)
            wall_time = perf_counter() - started
            if not isinstance(outcome, RolloutOutcome):
                raise TypeError("OOD rollouts must return initialization provenance")
            if (
                outcome.checkpoint_sha256 is not None
                and outcome.checkpoint_sha256 != candidate.checkpoint_sha256
            ):
                raise ValueError("OOD rollout policy differs from the recorded candidate")
            if reference_initializations is None:
                reference_initializations = outcome.initialization_indices
            elif outcome.initialization_indices != reference_initializations:
                raise ValueError("OOD methods were evaluated on different initializations")
            record = OODEvaluationRecord(
                run_id=run_id,
                held_out_suite=held_out_suite,
                task_id=task_id,
                seed=seed,
                method=method,
                adaptation_steps=candidate.adaptation_steps,
                rollout_count=rollout_count,
                successes=outcome.successes,
                checkpoint_sha256=candidate.checkpoint_sha256,
                wall_time_seconds=wall_time,
                initialization_indices=outcome.initialization_indices,
                rollout_seeds=outcome.rollout_seeds,
                source_training_suites=tuple(source_training_suites),
                alignment_checkpoint_sha256=alignment_checkpoint_sha256,
                adaptation_loss=candidate.adaptation_loss,
                adapter_bytes=candidate.adapter_bytes,
                evidence_bytes=candidate.evidence_bytes,
            )
            _ = record.success_rate
            records.append(record)
            if record_sink is not None:
                record_sink(record)
    return tuple(records)
