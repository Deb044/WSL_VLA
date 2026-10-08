#!/usr/bin/env python3
"""Extract provenance-locked Methodology 1 evidence from real LIBERO episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.alignment.evidence import masked_action_statistics
from wsl_vla.alignment.evidence_io import save_task_evidence
from wsl_vla.alignment.octo_evidence import (
    extract_frozen_octo_tokenizer_features,
    flatten_visual_token_groups,
)
from wsl_vla.contracts import TaskEvidence
from wsl_vla.data.libero import StrictLiberoHDF5
from wsl_vla.data.octo_batches import (
    ActionNormalization,
    collate_octo_examples,
    conform_batch_to_octo_example,
    iter_octo_examples,
)
from wsl_vla.experiments.protocol import (
    load_yaml,
    validate_reference_tasks,
    validate_research_config,
)
from wsl_vla.experiments.provenance import sha256_file
from wsl_vla.octo.bridge import load_research_octo


def _sample_episode_examples(
    episodes,
    *,
    instruction: str,
    normalization: ActionNormalization,
    window_size: int,
    action_horizon: int,
    image_size: tuple[int, int],
    maximum_examples: int,
):
    if maximum_examples < len(episodes):
        raise ValueError("maximum examples must allow at least one sample per training episode")
    base = maximum_examples // len(episodes)
    remainder = maximum_examples % len(episodes)
    sampled = []
    for episode_index, episode in enumerate(episodes):
        candidates = list(
            iter_octo_examples(
                (episode,),
                language_instruction=instruction,
                normalization=normalization,
                window_size=window_size,
                action_horizon=action_horizon,
                image_size=image_size,
                include_wrist=True,
            )
        )
        count = min(len(candidates), base + (episode_index < remainder))
        indices = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        sampled.extend(candidates[index] for index in indices)
    return sampled


def _padded_actions(episodes):
    maximum = max(len(episode.actions) for episode in episodes)
    action_dim = episodes[0].actions.shape[-1]
    actions = np.zeros((len(episodes), maximum, action_dim), dtype=np.float32)
    mask = np.zeros((len(episodes), maximum), dtype=bool)
    for index, episode in enumerate(episodes):
        count = len(episode.actions)
        actions[index, :count] = episode.actions
        mask[index, :count] = True
    return actions, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--maximum-examples", type=int, default=128)
    parser.add_argument("--maximum-visual-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    config = load_yaml(args.config)
    task_config = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(task_config)
    if args.suite not in task_config["suites"] or not 0 <= args.task_index < 10:
        raise ValueError("suite/task index is outside the locked reference protocol")
    if min(args.maximum_examples, args.maximum_visual_tokens, args.batch_size) <= 0:
        raise ValueError("sampling and batch sizes must be positive")
    instruction = task_config["suites"][args.suite][args.task_index]
    task_id = f"{args.suite}_{args.task_index}"

    dataset = StrictLiberoHDF5(
        args.data_file,
        require_wrist_camera=bool(config["data"]["require_wrist_camera"]),
        required_proprio_keys=tuple(config["data"]["proprio_keys"]),
    )
    split = dataset.split(seed=int(config["data"]["split_seed"]))
    episodes = tuple(dataset.iter_episodes(split.train))
    if not episodes:
        raise ValueError("training split contains no complete episodes")
    normalization = ActionNormalization.fit(episodes)

    bundle = load_research_octo(
        args.checkpoint,
        rank=int(config["adapter"]["rank"]),
        alpha=float(config["adapter"]["alpha"]),
        seed=0,
    )
    example = bundle.pretrained_model.example_batch
    primary_shape = np.asarray(example["observation"]["image_primary"]).shape
    window_size = int(primary_shape[1])
    image_size = (int(primary_shape[-3]), int(primary_shape[-2]))
    action_horizon = int(np.asarray(example["action"]).shape[-2])
    examples = _sample_episode_examples(
        episodes,
        instruction=instruction,
        normalization=normalization,
        window_size=window_size,
        action_horizon=action_horizon,
        image_size=image_size,
        maximum_examples=args.maximum_examples,
    )

    visual_parts = []
    language_parts = []
    for start in range(0, len(examples), args.batch_size):
        batch = collate_octo_examples(
            examples[start : start + args.batch_size],
            text_processor=bundle.pretrained_model.text_processor,
        )
        batch = conform_batch_to_octo_example(batch, example)
        visual, _, language = extract_frozen_octo_tokenizer_features(bundle, batch)
        visual_parts.append(visual)
        language_parts.append(language)
    visual_features, visual_mask = flatten_visual_token_groups(
        {
            "all_demonstrations": (
                np.concatenate(visual_parts, axis=0),
                np.ones(sum(len(part) for part in visual_parts), dtype=bool),
            )
        },
        maximum_tokens=args.maximum_visual_tokens,
    )
    if not all(np.allclose(language_parts[0], item, atol=1e-6) for item in language_parts[1:]):
        raise ValueError("identical task instructions produced inconsistent frozen language evidence")

    actions, action_mask = _padded_actions(episodes)
    action_statistics = masked_action_statistics(
        actions,
        action_mask,
        mean=normalization.mean,
        std=normalization.std,
    )
    preprocessing = {
        "schema_version": 1,
        "window_size": window_size,
        "action_horizon": action_horizon,
        "image_size": image_size,
        "maximum_examples": args.maximum_examples,
        "maximum_visual_tokens": args.maximum_visual_tokens,
        "split_seed": int(config["data"]["split_seed"]),
        "normalization_mean": normalization.mean.tolist(),
        "normalization_std": normalization.std.tolist(),
        "normalization_count": normalization.count,
    }
    preprocessing_version = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True).encode("utf-8")
    ).hexdigest()
    evidence = TaskEvidence(
        task_id=task_id,
        suite=args.suite,
        vision_features=visual_features,
        vision_mask=visual_mask,
        language_features=language_parts[0],
        action_statistics=action_statistics,
        episode_ids=tuple(episode.episode_id for episode in episodes),
        preprocessing_version=preprocessing_version,
        raw_feature_references={
            "dataset_sha256": sha256_file(args.data_file),
            "base_sha256": bundle.base_sha256,
            "base_revision": bundle.base_revision,
        },
        synthetic=False,
    )
    output = Path(args.output)
    save_task_evidence(evidence, output)
    manifest = {
        "schema_version": 1,
        "task_id": task_id,
        "suite": args.suite,
        "instruction": instruction,
        "evidence_sha256": sha256_file(output),
        "base_sha256": bundle.base_sha256,
        "base_revision": bundle.base_revision,
        "dataset_sha256": sha256_file(args.data_file),
        "train_episode_ids": list(evidence.episode_ids),
        "validation_episode_ids": list(split.validation),
        "test_episode_ids": list(split.test),
        "preprocessing": preprocessing,
        "preprocessing_version": preprocessing_version,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
