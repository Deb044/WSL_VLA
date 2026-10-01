#!/usr/bin/env python3
"""Train one official Octo-Small model-zoo run and save three late adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.adapters.packing import (
    PackedAdapterMetadata,
    pack_low_rank_adapter,
    save_packed_adapter,
)
from wsl_vla.alignment.evidence_io import load_task_evidence
from wsl_vla.data.libero import StrictLiberoHDF5, load_suite_manifest
from wsl_vla.data.octo_batches import (
    ActionNormalization,
    collate_octo_examples,
    conform_batch_to_octo_example,
    iter_octo_examples,
)
from wsl_vla.octo.bridge import (
    assert_zero_adapter_equivalence,
    build_adapter_spec_and_factors,
    load_research_octo,
)
from wsl_vla.octo.training import adapter_value_and_grad, initial_adapter_state
from wsl_vla.experiments.protocol import (
    load_yaml,
    validate_reference_tasks,
    validate_research_config,
)
from wsl_vla.experiments.provenance import sha256_file


def _epoch_batches(
    episodes,
    *,
    instruction,
    normalization,
    window_size,
    action_horizon,
    image_size,
    batch_size,
    rng,
    text_processor,
    example_batch,
):
    """Stream one shuffled epoch without materializing all image windows."""

    order = np.arange(len(episodes))
    rng.shuffle(order)
    pending = []
    for episode_index in order:
        examples = list(
            iter_octo_examples(
                (episodes[int(episode_index)],),
                language_instruction=instruction,
                normalization=normalization,
                window_size=window_size,
                action_horizon=action_horizon,
                image_size=image_size,
                include_wrist=True,
            )
        )
        rng.shuffle(examples)
        for example in examples:
            pending.append(example)
            if len(pending) == batch_size:
                batch = collate_octo_examples(pending, text_processor=text_processor)
                yield conform_batch_to_octo_example(batch, example_batch)
                pending = []
    if pending:
        batch = collate_octo_examples(pending, text_processor=text_processor)
        yield conform_batch_to_octo_example(batch, example_batch)


def _save_population_checkpoint(
    *,
    output_root: Path,
    suite: str,
    task_index: int,
    seed: int,
    step: int,
    fraction: float,
    bundle,
    adapter_state,
    token_width: int,
    evidence_path: Path,
    data_sha256: str,
    split: str,
):
    import jax

    task_id = f"{suite}_{task_index}"
    sample_dir = output_root / suite / task_id / f"seed_{seed}" / f"step_{step}"
    sample_dir.mkdir(parents=True, exist_ok=False)
    spec, factors = build_adapter_spec_and_factors(
        bundle,
        token_width=token_width,
        adapter_state=jax.device_get(adapter_state),
    )
    first_factor = next(iter(factors.values()))[0]
    packed = pack_low_rank_adapter(
        spec,
        factors,
        metadata=PackedAdapterMetadata(
            task_id=task_id,
            suite=suite,
            seed=seed,
            checkpoint_stage=step,
            checkpoint_fraction=fraction,
            source_dtype=str(np.asarray(first_factor).dtype),
        ),
    )
    adapter_path = sample_dir / "adapter.npz"
    save_packed_adapter(packed, adapter_path)
    shutil.copy2(evidence_path, sample_dir / "evidence.npz")
    evidence_manifest = evidence_path.with_suffix(".manifest.json")
    if evidence_manifest.is_file():
        shutil.copy2(evidence_manifest, sample_dir / "evidence.manifest.json")
    metadata = {
        "schema_version": 1,
        "synthetic": False,
        "task_id": task_id,
        "suite": suite,
        "task_index": task_index,
        "seed": seed,
        "checkpoint_stage": step,
        "checkpoint_fraction": fraction,
        "split": split,
        "base_sha256": bundle.base_sha256,
        "base_revision": bundle.base_revision,
        "dataset_sha256": data_sha256,
        "adapter_sha256": sha256_file(adapter_path),
        "evidence_sha256": sha256_file(sample_dir / "evidence.npz"),
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sample_dir, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--output-root", default="research_results/population")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    try:
        import jax
        import optax
    except ImportError as exc:
        raise RuntimeError("official zoo training requires the WSL2 JAX research environment") from exc

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    if args.suite not in tasks["suites"] or not 0 <= args.task_index < 10:
        raise ValueError("suite/task index is outside the locked reference protocol")
    if args.seed not in config["population"]["seeds"]:
        raise ValueError("seed is outside the locked three-seed population")
    instruction = tasks["suites"][args.suite][args.task_index]
    task_id = f"{args.suite}_{args.task_index}"
    evidence_path = Path(args.evidence)
    evidence = load_task_evidence(evidence_path)
    if evidence.task_id != task_id or evidence.suite != args.suite:
        raise ValueError("evidence identity does not match requested population run")

    data_root = Path(args.data_root or config["data"]["root"])
    task_files = load_suite_manifest(data_root / args.suite, tasks["suites"][args.suite], allow_missing=True)
    data_file = task_files[args.task_index]
    if data_file is None:
        raise FileNotFoundError(f"data file for task {args.task_index} is not downloaded in {data_root / args.suite}")
    if evidence.raw_feature_references.get("dataset_sha256") != sha256_file(data_file):
        raise ValueError("evidence was not extracted from the locked task dataset")
    dataset = StrictLiberoHDF5(
        data_file,
        require_wrist_camera=bool(config["data"]["require_wrist_camera"]),
        required_proprio_keys=tuple(config["data"]["proprio_keys"]),
    )
    episode_split = dataset.split(seed=int(config["data"]["split_seed"]))
    episodes = tuple(dataset.iter_episodes(episode_split.train))
    if tuple(episode.episode_id for episode in episodes) != evidence.episode_ids:
        raise ValueError("evidence training episodes differ from the locked episode split")
    normalization = ActionNormalization.fit(episodes)

    bundle = load_research_octo(
        args.checkpoint,
        rank=int(config["adapter"]["rank"]),
        alpha=float(config["adapter"]["alpha"]),
        seed=args.seed,
    )
    if evidence.raw_feature_references.get("base_sha256") != bundle.base_sha256:
        raise ValueError("evidence and zoo run use different Octo base checkpoints")
    assert_zero_adapter_equivalence(bundle)
    state = initial_adapter_state(bundle)
    accumulation_steps = int(config["continual_learning"]["gradient_accumulation_steps"])
    optimizer = optax.MultiSteps(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate)),
        every_k_schedule=accumulation_steps,
    )
    optimizer_state = optimizer.init(state)

    @jax.jit
    def train_step(adapter_state, opt_state, batch, key):
        (loss_and_metrics, gradients) = adapter_value_and_grad(
            bundle, adapter_state, batch, key
        )
        loss, metrics = loss_and_metrics
        updates, opt_state = optimizer.update(gradients, opt_state, adapter_state)
        adapter_state = optax.apply_updates(adapter_state, updates)
        return adapter_state, opt_state, loss, metrics

    example_batch = bundle.pretrained_model.example_batch
    primary_shape = np.asarray(example_batch["observation"]["image_primary"]).shape
    window_size = int(primary_shape[1])
    image_size = (int(primary_shape[-3]), int(primary_shape[-2]))
    action_horizon = int(np.asarray(example_batch["action"]).shape[-2])
    steps = args.steps or int(config["continual_learning"]["train_steps_per_task"])
    batch_size = args.batch_size or int(config["continual_learning"]["micro_batch_size"])
    if steps <= 0 or batch_size <= 0 or accumulation_steps <= 0 or args.learning_rate <= 0:
        raise ValueError("steps, batch size, and learning rate must be positive")
    fractions = tuple(float(value) for value in config["population"]["late_checkpoint_fractions"])
    save_steps = {max(1, int(round(steps * fraction))): fraction for fraction in fractions}
    if len(save_steps) != len(fractions):
        raise ValueError("training steps collapse distinct late-checkpoint fractions")

    rng = np.random.default_rng(args.seed)
    jax_key = jax.random.PRNGKey(args.seed)
    micro_step = 0
    update_step = 0
    history = []
    saved = []
    started = perf_counter()
    while update_step < steps:
        for batch in _epoch_batches(
            episodes,
            instruction=instruction,
            normalization=normalization,
            window_size=window_size,
            action_horizon=action_horizon,
            image_size=image_size,
            batch_size=batch_size,
            rng=rng,
            text_processor=bundle.pretrained_model.text_processor,
            example_batch=example_batch,
        ):
            if update_step >= steps:
                break
            micro_step += 1
            jax_key, step_key = jax.random.split(jax_key)
            state, optimizer_state, loss, metrics = train_step(
                state, optimizer_state, batch, step_key
            )
            if micro_step % accumulation_steps != 0:
                continue
            update_step += 1
            if update_step == 1 or update_step % 100 == 0 or update_step in save_steps:
                history.append(
                    {"step": update_step, "micro_step": micro_step, "loss": float(loss)}
                )
            if update_step in save_steps:
                sample_dir, metadata = _save_population_checkpoint(
                    output_root=Path(args.output_root),
                    suite=args.suite,
                    task_index=args.task_index,
                    seed=args.seed,
                    step=update_step,
                    fraction=save_steps[update_step],
                    bundle=bundle,
                    adapter_state=state,
                    token_width=int(config["adapter"]["token_width"]),
                    evidence_path=evidence_path,
                    data_sha256=sha256_file(data_file),
                    split="validation" if args.task_index in {8, 9} else "train",
                )
                saved.append(str(sample_dir))

    run_summary = {
        "schema_version": 1,
        "task_id": task_id,
        "suite": args.suite,
        "task_index": args.task_index,
        "seed": args.seed,
        "base_sha256": bundle.base_sha256,
        "base_revision": bundle.base_revision,
        "dataset_sha256": sha256_file(data_file),
        "steps": steps,
        "learning_rate": args.learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "wall_time_seconds": perf_counter() - started,
        "saved_samples": saved,
        "history": history,
    }
    summary_path = Path(args.output_root) / args.suite / task_id / f"seed_{args.seed}" / "run.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "summary": str(summary_path), "samples": saved}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
