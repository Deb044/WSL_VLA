"""Frozen official-Octo tokenizer evidence for Methodology 1."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def masked_pool_tokens(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool token features over every valid non-feature axis."""

    values = np.asarray(tokens)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim < 2 or valid.shape != values.shape[:-1]:
        raise ValueError("token mask must match every non-feature token axis")
    if not valid.any():
        raise ValueError("cannot pool an entirely masked token group")
    selected = values[valid]
    result = selected.mean(axis=0)
    if not np.isfinite(result).all():
        raise ValueError("pooled tokenizer evidence contains non-finite values")
    return result.astype(np.float32)


def flatten_visual_token_groups(
    groups: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    maximum_tokens: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine valid multi-view image-tokenizer outputs into one feature set."""

    if not groups:
        raise ValueError("Octo produced no visual tokenizer groups")
    widths = {np.asarray(tokens).shape[-1] for tokens, _ in groups.values()}
    if len(widths) != 1:
        raise ValueError("Octo visual tokenizer feature widths differ across views")
    rows = []
    for name in sorted(groups):
        tokens, mask = groups[name]
        values = np.asarray(tokens)
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != values.shape[:-1]:
            raise ValueError(f"visual mask shape differs for tokenizer {name}")
        rows.append(values[valid])
    features = np.concatenate(rows, axis=0)
    if maximum_tokens is not None:
        if maximum_tokens <= 0:
            raise ValueError("maximum_tokens must be positive")
        if len(features) > maximum_tokens:
            # Evenly-spaced deterministic sampling covers the whole demonstration
            # rather than biasing evidence toward its first frames.
            indices = np.linspace(0, len(features) - 1, maximum_tokens, dtype=np.int64)
            features = features[indices]
    if not len(features) or not np.isfinite(features).all():
        raise ValueError("visual tokenizer evidence must be non-empty and finite")
    return features.astype(np.float32), np.ones(len(features), dtype=bool)


def _tokenizer_outputs_method(module, observations, tasks):
    """Flax ``apply`` method that calls only frozen Octo tokenizer modules."""

    transformer = module.octo_transformer
    visual = {}
    image_names = {"primary", "secondary", "wrist"}
    for name, tokenizer in transformer.observation_tokenizers.items():
        output = tokenizer(observations, tasks, train=False)
        lowered = name.lower()
        if output is not None and ("image" in lowered or lowered in image_names):
            visual[name] = (output.tokens, output.mask)
    language = {}
    for name, tokenizer in transformer.task_tokenizers.items():
        output = tokenizer(observations, tasks, train=False)
        if output is not None and "language" in name.lower():
            language[name] = (output.tokens, output.mask)
    return {"visual": visual, "language": language}


def extract_frozen_octo_tokenizer_features(
    bundle: Any,
    batch: Mapping[str, Any],
    *,
    maximum_visual_tokens: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract visual token sets and masked-pooled language from official Octo.

    The pretrained parameter tree is used deliberately: evidence must not change
    with a task adapter or checkpoint from the model zoo.
    """

    required = {"observation", "task"}
    if required - set(batch):
        raise ValueError("Octo evidence batch requires observation and task")
    outputs = bundle.pretrained_model.module.apply(
        {"params": bundle.pretrained_model.params},
        batch["observation"],
        batch["task"],
        method=_tokenizer_outputs_method,
    )
    outputs = {
        family: {
            name: (np.asarray(tokens), np.asarray(mask))
            for name, (tokens, mask) in groups.items()
        }
        for family, groups in outputs.items()
    }
    visual_features, visual_mask = flatten_visual_token_groups(
        outputs["visual"], maximum_tokens=maximum_visual_tokens
    )
    if not outputs["language"]:
        raise ValueError("Octo produced no official language tokenizer group")
    language_widths = {
        tokens.shape[-1] for tokens, _ in outputs["language"].values()
    }
    if len(language_widths) != 1:
        raise ValueError("Octo language tokenizer feature widths differ")
    language_tokens = np.concatenate(
        [outputs["language"][name][0] for name in sorted(outputs["language"])], axis=-2
    )
    language_mask = np.concatenate(
        [outputs["language"][name][1] for name in sorted(outputs["language"])], axis=-1
    )
    language_features = masked_pool_tokens(language_tokens, language_mask)
    return visual_features, visual_mask, language_features
