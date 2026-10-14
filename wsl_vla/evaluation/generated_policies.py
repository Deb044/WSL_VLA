"""Policy conditions for evaluating reconstructed and evidence-generated adapters.

Every condition is deployed as rank-``r`` factors through the same low-rank
path as a zoo adapter (rank-matched deployment), and every condition is rolled
out on the same initial states with the same action-sampling seeds:

``frozen_base``
    the pinned checkpoint with zero adapters;
``zoo_adapter``
    the independently trained adapter of the evaluated task;
``reconstruction[<condition>]``
    that adapter after an autoencoder round trip;
``mapper[<condition>]`` / ``mapper_shell[<condition>]``
    an adapter generated from the task's evidence alone, without and with
    empirical-shell projection;
``nearest_neighbour_adapter``
    the trained adapter of the most similar *training* task (frozen evidence);
``training_mean_adapter``
    the task-balanced mean of the training tasks' updates.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..adapters.latent import factors_to_state
from ..adapters.packing import load_packed_adapter
from ..adapters.rank import project_updates_to_rank
from ..alignment.checkpoint import (
    compose_latent_sequence,
    decode_latent_sequence,
    encode_evidence_prompts,
    encode_weight_tokens,
    map_evidence_embeddings,
    rank_matched_updates,
)
from ..alignment.diagnostics import (
    TaskPopulation,
    mean_update_over_tasks,
    nearest_training_task,
)
from ..contracts import ROLLOUT_EPISODE_SCHEMA_VERSION, AdapterSpec, AlignmentCheckpoint
from ..experiments.manifest import ExperimentManifest, ExperimentPaths
from ..octo.training import initial_adapter_state
from .paired import classify_evidence, clustered_bootstrap, paired_differences


BASELINE_CONDITIONS = (
    "frozen_base",
    "zoo_adapter",
    "nearest_neighbour_adapter",
    "training_mean_adapter",
)
GENERATED_KINDS = ("reconstruction", "mapper", "mapper_shell")
PRIMARY_COMPARISONS = (
    ("zoo_adapter", "frozen_base"),
    ("mapper[aligned]", "mapper[reconstruction_only]"),
    ("mapper_shell[aligned]", "mapper_shell[reconstruction_only]"),
    ("mapper[aligned]", "frozen_base"),
    ("mapper[aligned]", "training_mean_adapter"),
    ("mapper[aligned]", "nearest_neighbour_adapter"),
    ("mapper[reconstruction_only]", "frozen_base"),
    ("reconstruction[aligned]", "zoo_adapter"),
    ("reconstruction[reconstruction_only]", "zoo_adapter"),
)


def condition_names(alignment_conditions: Sequence[str]) -> tuple[str, ...]:
    return (
        *BASELINE_CONDITIONS,
        *(f"{kind}[{condition}]" for kind in GENERATED_KINDS for condition in alignment_conditions),
    )


def reference_adapter(
    manifest: ExperimentManifest, paths: ExperimentPaths, identifier: str
) -> tuple[Any, dict]:
    """Packed reference adapter of a task (reference seed, final snapshot)."""

    sample = manifest.reference_sample(identifier)
    packed = load_packed_adapter(paths.sample_directory(sample) / "adapter.npz")
    return packed, {"sample_id": sample.sample_id, "run_id": sample.run_id}


def build_policy_states(
    *,
    bundle: Any,
    spec: AdapterSpec,
    manifest: ExperimentManifest,
    paths: ExperimentPaths,
    task: TaskPopulation,
    training: Mapping[str, TaskPopulation],
    checkpoints: Mapping[str, AlignmentCheckpoint],
    conditions: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Adapter state and deployment record for every requested condition."""

    known = condition_names(tuple(checkpoints))
    unknown = sorted(set(conditions) - set(known))
    if unknown:
        raise ValueError(f"unknown policy conditions {unknown}; available: {known}")
    reference, reference_identity = reference_adapter(manifest, paths, task.task_id)
    if reference.spec != spec:
        raise ValueError("reference zoo adapter uses a different adapter layout")
    policies: dict[str, dict[str, Any]] = {}

    def deploy(name: str, factors, **record) -> None:
        policies[name] = {
            "state": factors_to_state(bundle, spec, factors),
            "record": {"condition": name, "rank_matched": True, **record},
        }

    for name in conditions:
        if name == "frozen_base":
            policies[name] = {
                "state": initial_adapter_state(bundle),
                "record": {"condition": name, "rank_matched": True, "source": "zero adapters"},
            }
        elif name == "zoo_adapter":
            deploy(
                name,
                reference.factors,
                source=reference_identity["sample_id"],
                training_run=reference_identity["run_id"],
            )
        elif name == "nearest_neighbour_adapter":
            first = next(iter(checkpoints.values()))
            neighbour, similarity = nearest_training_task(
                first.standardization,
                task.evidence,
                {identifier: item.evidence for identifier, item in training.items()},
            )
            packed, identity = reference_adapter(manifest, paths, neighbour)
            deploy(
                name,
                packed.factors,
                source=identity["sample_id"],
                training_run=identity["run_id"],
                neighbour_task=neighbour,
                evidence_similarity=similarity,
            )
        elif name == "training_mean_adapter":
            projection = project_updates_to_rank(spec, mean_update_over_tasks(spec, training))
            deploy(
                name,
                projection.factors,
                source="task-balanced mean of training-task updates",
                training_tasks=sorted(training),
                rank_projection=projection.report(),
            )
        else:
            kind, _, condition = name.partition("[")
            checkpoint = checkpoints[condition.rstrip("]")]
            if kind == "reconstruction":
                latents = np.asarray(encode_weight_tokens(checkpoint, tokens=reference.tokens[None]))[0]
                decoded = np.asarray(decode_latent_sequence(checkpoint, latents))
                source = reference_identity["sample_id"]
            else:
                embeddings = encode_evidence_prompts(checkpoint, task.evidence)
                mapped = map_evidence_embeddings(
                    checkpoint, embeddings, shell_projection=(kind == "mapper_shell")
                )
                decoded = np.asarray(
                    decode_latent_sequence(checkpoint, compose_latent_sequence(checkpoint, mapped))
                )
                source = "task evidence only"
            projection = rank_matched_updates(checkpoint, decoded)
            deploy(
                name,
                projection.factors,
                source=source,
                alignment_condition=checkpoint.condition,
                alignment_checkpoint=checkpoint.metadata["directory"],
                rank_projection=projection.report(),
            )
    return policies


def episode_records(
    summary: Any,
    *,
    experiment: str,
    fold: str,
    alignment_seed: int | None,
    scope: str,
    condition: str,
    training_run: str,
) -> list[dict[str, Any]]:
    """One JSON-serializable record per rollout episode."""

    return [
        {
            "schema_version": ROLLOUT_EPISODE_SCHEMA_VERSION,
            "experiment": experiment,
            "fold": fold,
            "alignment_seed": alignment_seed,
            "scope": scope,
            "condition": condition,
            "training_run": training_run,
            "suite": summary.suite,
            "task_id": episode.task_id,
            "task_index": episode.task_index,
            "initialization_index": episode.initialization_index,
            "episode_seed": episode.episode_seed,
            "action_sampling_seed": episode.episode_seed,
            "success": bool(episode.success),
            "steps": int(episode.steps),
            "episode_return": float(episode.episode_return),
            "clipped_steps": int(episode.clipped_steps),
            "clipped_step_fraction": episode.clipped_step_fraction,
            "max_clip_excess": float(episode.max_clip_excess),
            "policy_checkpoint_sha256": summary.policy_checkpoint_sha256,
        }
        for episode in summary.episodes
    ]


def outcomes_by_condition(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """``condition -> task -> training run -> ordered episode outcomes``."""

    ordered = sorted(
        records,
        key=lambda item: (
            item["condition"],
            item["task_id"],
            item["training_run"],
            item["initialization_index"],
        ),
    )
    result: dict[str, dict[str, dict[str, list[float]]]] = {}
    for item in ordered:
        result.setdefault(item["condition"], {}).setdefault(item["task_id"], {}).setdefault(
            item["training_run"], []
        ).append(float(item["success"]))
    return result


def assert_paired_episodes(records: Sequence[Mapping[str, Any]]) -> None:
    """Every condition must hold the same (task, run, init state, seed) episodes."""

    identities: dict[str, set[tuple]] = {}
    for item in records:
        identities.setdefault(item["condition"], set()).add(
            (
                item["task_id"],
                item["training_run"],
                item["initialization_index"],
                item["episode_seed"],
            )
        )
    if not identities:
        raise ValueError("no rollout episodes to summarize")
    reference = next(iter(identities.values()))
    for condition, value in identities.items():
        if value != reference:
            raise ValueError(
                f"condition {condition} was not rolled out on the same initial states and seeds"
            )


def summarize_rollouts(
    records: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
    minimum_tasks: int,
    comparisons: Sequence[tuple[str, str]] = PRIMARY_COMPARISONS,
) -> dict[str, Any]:
    """Task-balanced success rates, paired differences, and clustered intervals."""

    assert_paired_episodes(records)
    outcomes = outcomes_by_condition(records)
    conditions = {}
    for name, values in outcomes.items():
        episodes = [item for item in records if item["condition"] == name]
        total_steps = sum(item["steps"] for item in episodes)
        clipped = sum(item["clipped_steps"] for item in episodes)
        conditions[name] = {
            "task_balanced_success_rate": clustered_bootstrap(
                values, resamples=resamples, confidence=confidence, seed=seed
            ),
            "per_task_success_rate": {
                task: float(np.mean([np.mean(run) for run in runs.values()]))
                for task, runs in values.items()
            },
            "episodes": len(episodes),
            "controller_clipping": {
                "steps": int(total_steps),
                "clipped_steps": int(clipped),
                "clipped_step_fraction": clipped / total_steps if total_steps else None,
            },
        }
    paired = {}
    for first, second in comparisons:
        if first not in outcomes or second not in outcomes:
            continue
        summary = clustered_bootstrap(
            paired_differences(outcomes[first], outcomes[second]),
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
        paired[f"{first} - {second}"] = {
            "paired_difference": summary,
            **classify_evidence(summary, higher_is_better=True, minimum_tasks=minimum_tasks),
        }
    return {"conditions": conditions, "paired_differences": paired}
