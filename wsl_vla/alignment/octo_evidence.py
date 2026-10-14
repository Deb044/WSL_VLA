"""Frozen official-Octo tokenizer evidence for Methodology 1.

Only *frozen* features are cached here, each tagged with the episode and
timestep it came from. How frames are aggregated into a task embedding and how
language and action evidence are projected is learned during alignment, so no
pooling over episodes or training of any kind happens at extraction time.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..contracts import TaskEvidence
from ..data.streams import TaskTrainingStream
from ..octo.compat import true_float32_matmul
from .evidence import action_chunk_evidence


def masked_pool_tokens(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool token features over the token axis, ignoring masked tokens."""

    values = np.asarray(tokens, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim < 2 or valid.shape != values.shape[:-1]:
        raise ValueError("token mask must match every non-feature token axis")
    counts = valid.sum(axis=-1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("cannot pool an entirely masked token group")
    pooled = (values * valid[..., None]).sum(axis=-2) / counts
    if not np.isfinite(pooled).all():
        raise ValueError("pooled tokenizer evidence contains non-finite values")
    return pooled.astype(np.float32)


def _tokenizer_outputs_method(module, observations, tasks):
    """Flax ``apply`` method that calls only frozen Octo tokenizer modules."""

    transformer = module.octo_transformer
    visual = {}
    for name, tokenizer in transformer.observation_tokenizers.items():
        if type(tokenizer).__name__ != "ImageTokenizer":
            continue
        output = tokenizer(observations, tasks, train=False)
        if output is not None:
            visual[name] = (output.tokens, output.mask)
    language = {}
    for name, tokenizer in transformer.task_tokenizers.items():
        if type(tokenizer).__name__ != "LanguageTokenizer":
            continue
        output = tokenizer(observations, tasks, train=False)
        if output is not None:
            language[name] = (output.tokens, output.mask)
    return {"visual": visual, "language": language}


def frozen_tokenizer_features(
    bundle: Any, batch: Mapping[str, Any], *, apply_fn: Any | None = None
) -> dict[str, Any]:
    """Per-example frozen features of one batch.

    Returns ``vision`` ``[batch, cameras * dim]`` - the masked token mean of the
    *last* (current) timestep of every image tokenizer, concatenated in sorted
    tokenizer order - plus ``language`` ``[batch, tokens, dim]`` and its mask.
    The pretrained parameter tree is used deliberately: evidence must not change
    with a task adapter or checkpoint from the model zoo.
    """

    required = {"observation", "task"}
    if required - set(batch):
        raise ValueError("Octo evidence batch requires observation and task")
    if apply_fn is None:
        apply_fn = make_tokenizer_apply(bundle)
    outputs = apply_fn(batch["observation"], batch["task"])
    if not outputs["visual"]:
        raise ValueError("Octo produced no visual tokenizer groups")
    if not outputs["language"]:
        raise ValueError("Octo produced no official language tokenizer group")
    timestep_valid = np.asarray(batch["observation"]["timestep_pad_mask"], dtype=bool)[:, -1]
    if not timestep_valid.all():
        raise ValueError("the current timestep of an evidence window cannot be padding")
    cameras = []
    for name in sorted(outputs["visual"]):
        tokens, mask = (np.asarray(value) for value in outputs["visual"][name])
        cameras.append(masked_pool_tokens(tokens[:, -1], mask[:, -1]))
    widths = {tokens.shape[-1] for tokens, _ in outputs["language"].values()}
    if len(widths) != 1:
        raise ValueError("Octo language tokenizer feature widths differ")
    names = sorted(outputs["language"])
    language = np.concatenate([np.asarray(outputs["language"][name][0]) for name in names], axis=-2)
    pad_mask = np.concatenate(
        [np.asarray(outputs["language"][name][1], dtype=bool) for name in names], axis=-1
    )
    attention = batch["task"].get("language_instruction")
    if isinstance(attention, Mapping) and "attention_mask" in attention and len(names) == 1:
        # Octo's own pad mask is per example; the tokenizer's attention mask
        # additionally removes padded token positions inside the instruction.
        pad_mask = pad_mask & np.asarray(attention["attention_mask"], dtype=bool)
    return {
        "vision": np.concatenate(cameras, axis=-1),
        "vision_cameras": sorted(outputs["visual"]),
        "language": language.astype(np.float32),
        "language_mask": pad_mask,
    }


def make_tokenizer_apply(bundle: Any):
    """Compiled frozen-tokenizer forward pass over the pinned pretrained tree.

    The pretrained parameters are placed on the device once and passed to the
    compiled function as an argument. Closing over them instead would embed the
    whole tree (including the T5 encoder) as program constants and make XLA
    spend minutes constant-folding them.
    """

    try:
        import jax
    except ImportError as exc:  # pragma: no cover - research environment only
        raise RuntimeError("frozen Octo evidence requires JAX") from exc
    module = bundle.pretrained_model.module
    transformer = bundle.pretrained_model.params["octo_transformer"]
    # Only the tokenizer subtrees are needed; the transformer blocks stay on host.
    params = jax.device_put(
        {
            "octo_transformer": {
                name: value
                for name, value in transformer.items()
                if name.startswith(("observation_tokenizers_", "task_tokenizers_"))
            }
        }
    )

    @jax.jit
    def apply(frozen, observations, tasks):
        return module.apply(
            {"params": frozen}, observations, tasks, method=_tokenizer_outputs_method
        )

    def run(observations, tasks):
        # True float32 matmuls: with the GPU default (TF32) the same instruction
        # encodes ~1e-3 differently from one batch shape to the next, and cached
        # evidence must not depend on how it was batched.
        with true_float32_matmul():
            return jax.device_get(apply(params, observations, tasks))

    return run


def evidence_frame_schedule(episode_length: int, frames_per_episode: int) -> np.ndarray:
    """Evenly spaced, de-duplicated timesteps covering one whole episode."""

    if episode_length <= 0 or frames_per_episode <= 0:
        raise ValueError("episode length and frames per episode must be positive")
    count = min(frames_per_episode, episode_length)
    return np.unique(np.round(np.linspace(0, episode_length - 1, count)).astype(np.int64))


def extract_task_evidence(
    bundle: Any,
    stream: TaskTrainingStream,
    *,
    suite: str,
    dataset_sha256: str,
    frames_per_episode: int,
    action_chunk_length: int,
    batch_size: int,
) -> TaskEvidence:
    """Cache frozen visual/language features and command-chunk summaries.

    ``stream`` must hold exactly the task's *training* episodes: evidence for a
    task is never computed from its validation or test demonstrations.
    """

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    apply_fn = make_tokenizer_apply(bundle)
    text_processor = bundle.pretrained_model.text_processor
    schedule = [
        (episode_index, int(timestep))
        for episode_index, episode in enumerate(stream.episodes)
        for timestep in evidence_frame_schedule(len(episode.actions), frames_per_episode)
    ]
    vision_parts = []
    language = None
    language_mask = None
    cameras = None
    for start in range(0, len(schedule), batch_size):
        chunk = schedule[start : start + batch_size]
        batch = stream.collate([stream.example(*item) for item in chunk], text_processor)
        features = frozen_tokenizer_features(bundle, batch, apply_fn=apply_fn)
        vision_parts.append(features["vision"])
        cameras = features["vision_cameras"]
        for row, mask in zip(features["language"], features["language_mask"]):
            if language is None:
                language, language_mask = row, mask
            elif not np.array_equal(mask, language_mask) or not np.allclose(
                row[mask], language[mask], rtol=1e-4, atol=1e-4
            ):
                raise ValueError(
                    "identical task instructions produced inconsistent frozen language evidence"
                )
    chunk_features, chunk_mask = action_chunk_evidence(
        [episode.actions for episode in stream.episodes], chunk_length=action_chunk_length
    )
    evidence = TaskEvidence(
        task_id=stream.task_id,
        suite=suite,
        instruction=stream.instruction,
        episode_ids=stream.episode_ids,
        vision_features=np.concatenate(vision_parts).astype(np.float32),
        vision_episode_index=np.asarray([item[0] for item in schedule], dtype=np.int32),
        vision_timestep=np.asarray([item[1] for item in schedule], dtype=np.int32),
        language_features=np.asarray(language, dtype=np.float32),
        language_mask=np.asarray(language_mask, dtype=bool),
        action_chunk_features=chunk_features,
        action_chunk_mask=chunk_mask,
        feature_extractor_sha256=dict(bundle.feature_extractor_sha256),
        preprocessing_sha256=bundle.preprocessing.sha256(),
        action_convention_sha256=bundle.action_convention.sha256(),
        base_sha256=bundle.base_sha256,
        base_revision=bundle.base_revision,
        architecture_sha256=bundle.architecture_sha256,
        dataset_sha256=dataset_sha256,
        extraction={
            "frames_per_episode": int(frames_per_episode),
            "action_chunk_length": int(action_chunk_length),
            "vision_cameras": list(cameras or ()),
            "vision_pooling": "masked token mean of the current timestep, per camera",
            "language_mask": "tokenizer attention mask AND Octo task pad mask",
            "action_summary_fields": "mean/std of commands and command differences 1-3",
        },
        synthetic=False,
    )
    evidence.validate_for_research()
    return evidence
