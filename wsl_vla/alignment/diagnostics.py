"""Paired offline diagnostics for the two primary alignment conditions.

For a set of evaluated tasks (the fold's validation tasks during development,
or its held-out suite in the final evaluation) this module reports, per
condition and task-balanced:

* autoencoder reconstruction error of real zoo adapters;
* mapped decoded-update error (evidence -> mapper -> decoder), before and
  after empirical-shell projection, and after rank-matched projection;
* rank-projection error;
* latent prediction error relative to a training-mean predictor;
* evidence-to-mapped-latent retrieval with an explicit chance level.

It also reports two baselines that involve no learned weight space (the
training-mean update and the nearest-neighbour training task's update), and a
direct weight/evidence cosine retrieval that is labelled descriptive: the
control was never trained to align those coordinates, so it is not a fair
standalone advantage test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..adapters.packing import load_packed_adapter, unpack_tokens
from ..contracts import COMPONENT_NAMES, AdapterSpec, AlignmentCheckpoint, TaskEvidence
from ..evaluation.paired import classify_evidence, clustered_bootstrap, paired_differences
from ..experiments.manifest import ExperimentManifest, ExperimentPaths, SampleIdentity
from .archive import _load_fold_evidence, validate_population
from .checkpoint import (
    assert_paired_checkpoints,
    compose_latent_sequence,
    decode_latent_sequence,
    decoded_updates,
    encode_evidence_prompts,
    encode_weight_tokens,
    map_evidence_embeddings,
    rank_matched_updates,
    split_latent_sequence,
)
from .evaluation import (
    INTEGRITY_PASS,
    assert_finite_report,
    component_update_error,
    gallery_retrieval,
    pooled_token_embedding,
    relative_to_training_mean,
    summarize_task_metric,
)
from .evidence import EvidenceStandardization


DIAGNOSTICS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TaskPopulation:
    """Verified zoo samples and evidence of one task."""

    task_id: str
    evidence: TaskEvidence
    samples: tuple[SampleIdentity, ...]
    tokens: np.ndarray

    def mean_update(self, spec: AdapterSpec) -> dict[str, np.ndarray]:
        return unpack_tokens(spec, self.tokens.mean(axis=0))


class PopulationIndex:
    """Verified zoo samples and evidence of a set of tasks, loaded one task at a time.

    Hashes and identities are re-checked once at construction. Token arrays are
    read only on demand, so evaluating a whole suite never holds more than one
    task's adapters in host memory.
    """

    def __init__(
        self, manifest: ExperimentManifest, paths: ExperimentPaths, tasks: Sequence[str]
    ) -> None:
        self.tasks = tuple(tasks)
        population = validate_population(manifest, paths, samples=manifest.samples(self.tasks))
        self.evidence = _load_fold_evidence(manifest, paths, self.tasks, population)["records"]
        self._records: dict[str, list[tuple[SampleIdentity, Any]]] = {task: [] for task in self.tasks}
        for sample, directory, _ in population["records"]:
            self._records[sample.task_id].append((sample, directory))
        self.spec: AdapterSpec | None = None

    def _packed(self, directory):
        packed = load_packed_adapter(directory / "adapter.npz")
        if self.spec is None:
            self.spec = packed.spec
        elif packed.spec != self.spec:
            raise ValueError("zoo samples have incompatible adapter layouts")
        return packed

    def load(self, task: str) -> TaskPopulation:
        members = self._records[task]
        return TaskPopulation(
            task_id=task,
            evidence=self.evidence[task],
            samples=tuple(sample for sample, _ in members),
            tokens=np.stack([self._packed(directory).tokens for _, directory in members]),
        )

    def load_mean(self, task: str) -> TaskPopulation:
        """The task with its samples reduced to their mean tokens (one row)."""

        members = self._records[task]
        total = None
        for _, directory in members:
            tokens = self._packed(directory).tokens.astype(np.float64)
            total = tokens if total is None else total + tokens
        return TaskPopulation(
            task_id=task,
            evidence=self.evidence[task],
            samples=tuple(sample for sample, _ in members),
            tokens=(total / len(members)).astype(np.float32)[None],
        )

    def load_all_means(self) -> dict[str, TaskPopulation]:
        return {task: self.load_mean(task) for task in self.tasks}


def load_task_populations(
    manifest: ExperimentManifest,
    paths: ExperimentPaths,
    tasks: Sequence[str],
) -> tuple[dict[str, TaskPopulation], AdapterSpec]:
    """Eagerly load every sample of ``tasks`` (small populations and tests)."""

    index = PopulationIndex(manifest, paths, tasks)
    populations = {task: index.load(task) for task in index.tasks}
    assert index.spec is not None
    return populations, index.spec


def frozen_evidence_signature(
    standardization: EvidenceStandardization, evidence: TaskEvidence
) -> np.ndarray:
    """Checkpoint-independent task signature for nearest-neighbour retrieval.

    Standardized frozen features are mean-pooled per modality, each modality is
    L2-normalized, and the three blocks are concatenated, so the cosine between
    two signatures is the mean of the three per-modality cosines.
    """

    arrays = standardization.apply(evidence)
    blocks = []
    for name in COMPONENT_NAMES:
        values = arrays[name].reshape(-1, arrays[name].shape[-1])
        mask = arrays[f"{name}_mask"].reshape(-1)
        pooled = values[mask].mean(axis=0)
        blocks.append(pooled / max(float(np.linalg.norm(pooled)), 1e-12))
    return np.concatenate(blocks) / np.sqrt(len(blocks))


def nearest_training_task(
    standardization: EvidenceStandardization,
    query: TaskEvidence,
    training: Mapping[str, TaskEvidence],
) -> tuple[str, float]:
    """Most similar *training* task by frozen evidence (ties: first in sorted order)."""

    signature = frozen_evidence_signature(standardization, query)
    best, best_similarity = "", -np.inf
    for task in sorted(training):
        similarity = float(signature @ frozen_evidence_signature(standardization, training[task]))
        if similarity > best_similarity:
            best, best_similarity = task, similarity
    return best, best_similarity


def mean_update_over_tasks(
    spec: AdapterSpec, populations: Mapping[str, TaskPopulation]
) -> dict[str, np.ndarray]:
    """Task-balanced mean effective update (each task counts once)."""

    means = [population.mean_update(spec) for population in populations.values()]
    return {
        entry.parameter_path: np.mean([item[entry.parameter_path] for item in means], axis=0)
        for entry in spec.entries
    }


def _mapped_report(checkpoint, mapped, target_update, task_latents, training_mean) -> dict:
    sequence = compose_latent_sequence(checkpoint, mapped)
    tokens = np.asarray(decode_latent_sequence(checkpoint, sequence))
    unrestricted = decoded_updates(checkpoint, tokens)
    projection = rank_matched_updates(checkpoint, tokens)
    spec = checkpoint.adapter_spec
    latent = {
        name: relative_to_training_mean(
            np.asarray(mapped[name]), task_latents[name], training_mean[name]
        )
        for name in COMPONENT_NAMES
    }
    defined = [
        latent[name]["relative_to_training_mean"]
        for name in COMPONENT_NAMES
        if latent[name]["relative_to_training_mean"] is not None
    ]
    return {
        "unrestricted_update_error": component_update_error(spec, unrestricted, target_update),
        "rank_matched_update_error": component_update_error(spec, projection.updates, target_update),
        "rank_projection": projection.report(),
        "latent_prediction": latent,
        "latent_relative_to_training_mean": float(np.mean(defined)) if defined else None,
        "sequence": np.asarray(sequence),
    }


class ConditionDiagnostics:
    """Accumulates one condition's per-task diagnostics, one task at a time."""

    def __init__(self, checkpoint: AlignmentCheckpoint) -> None:
        self.checkpoint = checkpoint
        bank = checkpoint.metadata["task_bank"]
        self.training_mean = {
            name: bank[f"train_mean_latents_{name}"].mean(axis=0) for name in COMPONENT_NAMES
        }
        self.training_mean_sequence = np.asarray(
            compose_latent_sequence(checkpoint, self.training_mean)
        )
        self.per_task: dict[str, Any] = {}
        self.reconstruction = {name: {} for name in (*COMPONENT_NAMES, "macro")}
        self.reconstruction_rank: dict[str, list] = {}
        self.mapped = {
            variant: {"unrestricted": {}, "rank_matched": {}, "rank_projection": {}, "latent": {}}
            for variant in ("mapped", "mapped_shell")
        }
        self.queries = {"mapped": [], "mapped_shell": []}
        self.gallery: list[np.ndarray] = []
        self.pooled = {name: [] for name in COMPONENT_NAMES}
        self.evidence_embeddings = {name: [] for name in COMPONENT_NAMES}
        self.labels: list[str] = []

    def add_task(self, population: TaskPopulation) -> None:
        checkpoint = self.checkpoint
        spec = checkpoint.adapter_spec
        task = population.task_id
        latents = np.asarray(encode_weight_tokens(checkpoint, tokens=population.tokens))
        sample_reports = []
        rank_errors = []
        for index in range(len(latents)):
            tokens = np.asarray(decode_latent_sequence(checkpoint, latents[index]))
            truth = unpack_tokens(spec, population.tokens[index])
            sample_reports.append(
                component_update_error(spec, decoded_updates(checkpoint, tokens), truth)
            )
            rank_errors.append(rank_matched_updates(checkpoint, tokens).relative_error)
        for name in self.reconstruction:
            self.reconstruction[name][task] = [report[name] for report in sample_reports]
        self.reconstruction_rank[task] = rank_errors
        task_latent_sequence = latents.mean(axis=0)
        task_latents = {
            name: np.asarray(value)
            for name, value in split_latent_sequence(checkpoint, task_latent_sequence).items()
        }
        self.gallery.append(task_latent_sequence)
        embeddings = encode_evidence_prompts(checkpoint, population.evidence)
        for name in COMPONENT_NAMES:
            self.evidence_embeddings[name].append(embeddings[name])
            positions = checkpoint.layout.component_positions(name)
            self.pooled[name].append(
                np.mean([pooled_token_embedding(sample[positions]) for sample in latents], axis=0)
            )
        target_update = population.mean_update(spec)
        task_report = {}
        for variant, shell in (("mapped", False), ("mapped_shell", True)):
            mapped = map_evidence_embeddings(checkpoint, embeddings, shell_projection=shell)
            report = _mapped_report(
                checkpoint, mapped, target_update, task_latents, self.training_mean
            )
            self.queries[variant].append(report.pop("sequence"))
            metrics = self.mapped[variant]
            metrics["unrestricted"][task] = [report["unrestricted_update_error"]["macro"]]
            metrics["rank_matched"][task] = [report["rank_matched_update_error"]["macro"]]
            metrics["rank_projection"][task] = [report["rank_projection"]["relative_error"]]
            metrics["latent"][task] = [report["latent_relative_to_training_mean"]]
            task_report[variant] = report
        self.per_task[task] = task_report
        self.labels.append(task)

    def report(self) -> dict[str, Any]:
        labels = self.labels
        retrieval = {
            variant: gallery_retrieval(
                np.stack(self.queries[variant]),
                np.stack(self.gallery),
                labels,
                labels,
                center=self.training_mean_sequence,
            )
            for variant in self.queries
        }
        direct = {
            name: gallery_retrieval(
                np.stack(self.evidence_embeddings[name]), np.stack(self.pooled[name]), labels, labels
            )
            for name in COMPONENT_NAMES
        }
        return {
            "condition": self.checkpoint.condition,
            "reconstruction_error": {
                name: summarize_task_metric(values) for name, values in self.reconstruction.items()
            },
            "reconstruction_rank_projection_error": summarize_task_metric(self.reconstruction_rank),
            "mapping": {
                variant: {
                    "unrestricted_update_error": summarize_task_metric(metrics["unrestricted"]),
                    "rank_matched_update_error": summarize_task_metric(metrics["rank_matched"]),
                    "rank_projection_error": summarize_task_metric(metrics["rank_projection"]),
                    "latent_error_relative_to_training_mean": summarize_task_metric(metrics["latent"]),
                    "evidence_to_mapped_latent_retrieval": retrieval[variant],
                }
                for variant, metrics in self.mapped.items()
            },
            "direct_weight_evidence_cosine_retrieval": {
                "note": (
                    "Descriptive alignment diagnostic only. Unaligned coordinates have no "
                    "reason to match evidence coordinates, so this is not a fair standalone "
                    "advantage test against the reconstruction-only control."
                ),
                "per_modality": direct,
            },
            "per_task": self.per_task,
        }


def evaluate_condition(
    checkpoint: AlignmentCheckpoint,
    populations: Mapping[str, TaskPopulation],
) -> dict[str, Any]:
    """Every per-task diagnostic of one alignment condition."""

    accumulator = ConditionDiagnostics(checkpoint)
    for population in populations.values():
        accumulator.add_task(population)
    return accumulator.report()


def _per_task_values(summary: Mapping[str, Any], run: str) -> dict[str, dict[str, list[float]]]:
    return {task: {run: [value]} for task, value in summary["per_task"].items()}


def compare_conditions(
    aligned: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    run: str,
    minimum_tasks: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Paired, task-clustered comparison of the aligned condition with its control."""

    comparisons = {}
    plans = []
    for variant in ("mapped", "mapped_shell"):
        for metric in ("rank_matched_update_error", "latent_error_relative_to_training_mean"):
            plans.append(
                (
                    f"{variant}.{metric}",
                    _per_task_values(aligned["mapping"][variant][metric], run),
                    _per_task_values(control["mapping"][variant][metric], run),
                    False,
                )
            )
        key = "evidence_to_mapped_latent_retrieval"
        first = aligned["mapping"][variant][key]["per_query_correct"]
        second = control["mapping"][variant][key]["per_query_correct"]
        plans.append(
            (
                f"{variant}.{key}.top1",
                {task: {run: [float(hit)]} for task, hit in first.items()},
                {task: {run: [float(hit)]} for task, hit in second.items()},
                True,
            )
        )
    for name, first, second, higher_is_better in plans:
        shared = sorted(set(first) & set(second))
        difference = paired_differences(
            {task: first[task] for task in shared}, {task: second[task] for task in shared}
        )
        summary = clustered_bootstrap(
            difference, resamples=resamples, confidence=confidence, seed=seed
        )
        comparisons[name] = {
            "definition": "aligned minus reconstruction_only, paired by task",
            "higher_is_better": higher_is_better,
            "undefined_tasks": sorted((set(first) | set(second)) - set(shared)),
            "paired_difference": summary,
            **classify_evidence(
                summary, higher_is_better=higher_is_better, minimum_tasks=minimum_tasks
            ),
        }
    return comparisons


def run_pair_diagnostics(
    *,
    manifest: ExperimentManifest,
    paths: ExperimentPaths,
    aligned: AlignmentCheckpoint,
    control: AlignmentCheckpoint,
    evaluated_tasks: Sequence[str],
    scope: str,
    minimum_tasks: int,
    resamples: int,
    confidence: float,
) -> dict[str, Any]:
    """Full paired report. Raises only for integrity failures."""

    assert_paired_checkpoints(aligned, control)
    if scope not in {"validation", "held_out"}:
        raise ValueError("scope must be validation or held_out")
    train_tasks = tuple(aligned.metadata["train_task_ids"])
    if scope == "held_out" and set(evaluated_tasks) & set(train_tasks):
        raise ValueError("held-out evaluation tasks overlap the alignment training tasks")
    evaluated = PopulationIndex(manifest, paths, evaluated_tasks)
    training = PopulationIndex(manifest, paths, train_tasks).load_all_means()
    spec = aligned.adapter_spec
    training_evidence = {name: item.evidence for name, item in training.items()}
    training_mean_update = None
    baseline_mean = {}
    baseline_neighbour = {}
    neighbours = {}
    accumulators = {
        "aligned": ConditionDiagnostics(aligned),
        "reconstruction_only": ConditionDiagnostics(control),
    }
    for task in evaluated.tasks:
        population = evaluated.load(task)
        if evaluated.spec != spec:
            raise ValueError("zoo samples and alignment checkpoints use different adapter layouts")
        if training_mean_update is None:
            training_mean_update = mean_update_over_tasks(spec, training)
        target = population.mean_update(spec)
        baseline_mean[task] = [component_update_error(spec, training_mean_update, target)["macro"]]
        neighbour, similarity = nearest_training_task(
            aligned.standardization, population.evidence, training_evidence
        )
        neighbours[task] = {"training_task": neighbour, "similarity": similarity}
        baseline_neighbour[task] = [
            component_update_error(spec, training[neighbour].mean_update(spec), target)["macro"]
        ]
        for accumulator in accumulators.values():
            accumulator.add_task(population)
    reports = {name: accumulator.report() for name, accumulator in accumulators.items()}
    comparisons = compare_conditions(
        reports["aligned"],
        reports["reconstruction_only"],
        run=f"alignment_seed_{aligned.config['seed']}",
        minimum_tasks=minimum_tasks,
        resamples=resamples,
        confidence=confidence,
        seed=int(aligned.config["seed"]),
    )
    report = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "experiment": manifest.name,
        "profile": manifest.profile,
        "publication_evidence": manifest.publication_evidence and scope == "held_out",
        "integrity_only": bool(aligned.metadata["integrity_only"]),
        "fold": aligned.metadata["fold"],
        "alignment_seed": int(aligned.config["seed"]),
        "scope": scope,
        "evaluated_tasks": list(evaluated_tasks),
        "training_tasks": list(train_tasks),
        "integrity": {
            "status": INTEGRITY_PASS,
            "checks": [
                "conditions share archive, fold, seed, layout, scales, standardization",
                "conditions share initialization, batch schedule, and optimizer exposure",
                "configurations differ only in the contrastive term",
                "zoo sample and evidence hashes re-verified",
                "all reported values finite or explicitly undefined",
            ],
        },
        "conditions": reports,
        "baselines": {
            "training_mean_update_error": summarize_task_metric(baseline_mean),
            "nearest_neighbour_update_error": summarize_task_metric(baseline_neighbour),
            "nearest_neighbours": neighbours,
            "note": "Baselines use true zoo updates and frozen evidence only; no learned weight space.",
        },
        "comparisons": comparisons,
        "interpretation": (
            "Infrastructure passes when integrity passes. Each comparison is labelled "
            "inconclusive, supports_first_condition (aligned), or negative_finding; none "
            "of these labels changes the exit status."
        ),
    }
    assert_finite_report({k: v for k, v in report.items() if k != "conditions"})
    assert_finite_report(
        {name: {k: v for k, v in value.items() if k != "per_task"} for name, value in reports.items()}
    )
    return report
