"""A toy-sized official Octo model for bridge, policy, and zoo-trainer tests.

The model is assembled from Octo's own classes - ``ImageTokenizer``,
``LanguageTokenizer`` (with a one-layer random T5 encoder), ``BlockTransformer``,
``DiffusionActionHead`` - through ``OctoModel.from_config``, exactly as the
pinned checkpoint is, only with tiny dimensions and random weights. Tests
therefore exercise the real patched graph, loss, and sampler without the 547 MB
checkpoint or network access.

Importing this module requires one of the research environments; test modules
guard it with ``pytest.importorskip("octo")``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tests.fakes import SOURCE, FakeTextProcessor, episodes
from wsl_vla.contracts import LIBERO_ACTION_CONVENTION
from wsl_vla.data.streams import TaskTrainingStream, task_stream_from_bundle
from wsl_vla.experiments.manifest import ZooBudget
from wsl_vla.octo.bridge import (
    ResearchOctoBundle,
    build_adapter_spec_and_factors,
    build_research_bundle,
    restore_official_octo,
)
from wsl_vla.octo.zoo import ZooRunIdentity


TOKEN_DIM = 16
WINDOW = 2
ACTION_HORIZON = 3
PROPRIO_DIM = 3
PRIMARY_SIZE = 16
WRIST_SIZE = 8
PATCH = 8
LANGUAGE_TOKENS = FakeTextProcessor.length
TOKEN_WIDTH = 8
SEQUENCE_WINDOW = 16
INSTRUCTION = "put the bowl on the plate"

TINY_PREPROCESSING = {
    "source_camera_height": SOURCE[0],
    "source_camera_width": SOURCE[1],
    "demonstration_image_convention": "opengl",
    "image_transform": "flip_vertical",
    "simulator_alignment": "identity",
    "executed_actions_per_plan": 1,
}

# Token counts per group, in the order Octo assembles one timestep.
PRIMARY_TOKENS = (PRIMARY_SIZE // PATCH) ** 2
WRIST_TOKENS = (WRIST_SIZE // PATCH) ** 2


def write_tiny_t5(directory: Path) -> str:
    """Write a one-layer T5 configuration Octo's LanguageTokenizer can load offline.

    The directory name contains ``t5`` because the tokenizer selects its encoder
    class from the model name.
    """

    from transformers import T5Config

    target = Path(directory) / "tiny-t5"
    target.mkdir(parents=True, exist_ok=True)
    T5Config(
        vocab_size=128,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        relative_attention_num_buckets=8,
        dropout_rate=0.0,
        feed_forward_proj="relu",
    ).save_pretrained(target)
    return str(target)


def tiny_model_config(t5_directory: str, *, proprio: bool = False, layers: int = 2) -> dict:
    """Model section mirroring the octo-small-1.5 structure at toy size."""

    from octo.model.components.action_heads import DiffusionActionHead
    from octo.model.components.tokenizers import (
        ImageTokenizer,
        LanguageTokenizer,
        LowdimObsTokenizer,
    )
    from octo.model.components.vit_encoders import PatchEncoder
    from octo.utils.spec import ModuleSpec

    def camera(key: str) -> dict:
        return ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=[key],
            task_stack_keys=[key],
            encoder=ModuleSpec.create(PatchEncoder, patch_size=PATCH, num_features=TOKEN_DIM),
        )

    observation_tokenizers = {"primary": camera("image_primary"), "wrist": camera("image_wrist")}
    if proprio:
        observation_tokenizers["proprio"] = ModuleSpec.create(
            LowdimObsTokenizer, obs_keys=["proprio"], discretize=False
        )
    return {
        "observation_tokenizers": observation_tokenizers,
        "task_tokenizers": {
            "language": ModuleSpec.create(
                LanguageTokenizer, encoder=t5_directory, finetune_encoder=False
            )
        },
        "heads": {
            "action": ModuleSpec.create(
                DiffusionActionHead,
                readout_key="readout_action",
                use_map=False,
                action_horizon=ACTION_HORIZON,
                action_dim=LIBERO_ACTION_CONVENTION.action_dim,
                max_action=5.0,
                time_dim=8,
                num_blocks=1,
                hidden_dim=16,
                diffusion_steps=3,
            )
        },
        "readouts": {"action": 1},
        "transformer_kwargs": {
            "num_layers": layers,
            "mlp_dim": 32,
            "num_attention_heads": 2,
            "dropout_rate": 0.0,
            "attention_dropout_rate": 0.0,
            "add_position_embedding": False,
        },
        "token_embedding_size": TOKEN_DIM,
        "max_horizon": 4,
        "repeat_task_tokens": True,
        "use_correct_attention": True,
    }


def tiny_example_batch(*, proprio: bool = False, seed: int = 0) -> dict:
    """Checkpoint-style example batch (leading batch axis of one) with real content."""

    rng = np.random.default_rng(seed)
    image = lambda size, *lead: rng.integers(  # noqa: E731
        0, 255, size=(*lead, size, size, 3), dtype=np.uint8
    )
    language = FakeTextProcessor().encode([INSTRUCTION])
    observation: dict[str, Any] = {
        "image_primary": image(PRIMARY_SIZE, 1, WINDOW),
        "image_wrist": image(WRIST_SIZE, 1, WINDOW),
        "timestep_pad_mask": np.ones((1, WINDOW), dtype=bool),
        "timestep": np.arange(WINDOW, dtype=np.int32)[None],
        "task_completed": np.zeros((1, WINDOW, ACTION_HORIZON), dtype=bool),
        "pad_mask_dict": {
            "image_primary": np.ones((1, WINDOW), dtype=bool),
            "image_wrist": np.ones((1, WINDOW), dtype=bool),
            "timestep": np.ones((1, WINDOW), dtype=bool),
        },
    }
    if proprio:
        observation["proprio"] = rng.normal(size=(1, WINDOW, PROPRIO_DIM)).astype(np.float32)
        observation["pad_mask_dict"]["proprio"] = np.ones((1, WINDOW), dtype=bool)
    return {
        "observation": observation,
        "task": {
            "image_primary": np.zeros((1, PRIMARY_SIZE, PRIMARY_SIZE, 3), dtype=np.uint8),
            "image_wrist": np.zeros((1, WRIST_SIZE, WRIST_SIZE, 3), dtype=np.uint8),
            "language_instruction": language,
            "pad_mask_dict": {
                "image_primary": np.zeros((1,), dtype=bool),
                "image_wrist": np.zeros((1,), dtype=bool),
                "language_instruction": np.ones((1,), dtype=bool),
            },
        },
        "action": np.zeros(
            (1, WINDOW, ACTION_HORIZON, LIBERO_ACTION_CONVENTION.action_dim), dtype=np.float32
        ),
        "action_pad_mask": np.ones(
            (1, WINDOW, ACTION_HORIZON, LIBERO_ACTION_CONVENTION.action_dim), dtype=bool
        ),
    }


def build_tiny_pretrained(
    t5_directory: str, *, seed: int = 0, proprio: bool = False, layers: int = 2
) -> Any:
    """A randomly initialized official ``OctoModel`` standing in for the checkpoint."""

    import jax
    from octo.model.octo_model import OctoModel

    config = {
        "model": tiny_model_config(t5_directory, proprio=proprio, layers=layers),
        "text_processor": {"name": "FakeTextProcessor", "length": LANGUAGE_TOKENS},
    }
    # The stand-in for ``OctoModel.load_pretrained``: it must see official Octo.
    restore_official_octo()
    return OctoModel.from_config(
        config,
        tiny_example_batch(proprio=proprio),
        FakeTextProcessor(),
        rng=jax.random.PRNGKey(seed),
        dataset_statistics=None,
    )


def build_tiny_bundle(
    t5_directory: str,
    *,
    rank: int = 2,
    alpha: float = 4.0,
    seed: int = 0,
    layers: int = 2,
    preprocessing: dict | None = None,
) -> ResearchOctoBundle:
    return build_research_bundle(
        build_tiny_pretrained(t5_directory, seed=seed, layers=layers),
        base_revision="tiny-octo-test",
        rank=rank,
        alpha=alpha,
        seed=seed,
        preprocessing=TINY_PREPROCESSING if preprocessing is None else preprocessing,
    )


def tiny_stream(
    bundle: ResearchOctoBundle,
    *,
    count: int = 4,
    seed: int = 0,
    task_id: str = "libero_spatial_0",
    instruction: str = INSTRUCTION,
) -> TaskTrainingStream:
    return task_stream_from_bundle(
        bundle, task_id=task_id, instruction=instruction, episodes=episodes(count, seed=seed)
    )


def tiny_budget(
    *,
    optimizer_updates: int = 4,
    snapshot_updates: tuple[int, ...] = (2, 4),
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-2,
    held_out_loss_every: int = 2,
) -> ZooBudget:
    return ZooBudget(
        optimizer_updates=optimizer_updates,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        snapshot_updates=snapshot_updates,
        held_out_loss_every=held_out_loss_every,
        held_out_loss_batches=2,
        frozen=True,
    )


def tiny_identity(
    bundle: ResearchOctoBundle,
    stream: TaskTrainingStream,
    held_out: TaskTrainingStream,
    budget: ZooBudget,
    *,
    seed: int = 17,
    dataset_sha256: str = "d" * 64,
) -> ZooRunIdentity:
    spec, _ = build_adapter_spec_and_factors(bundle, token_width=TOKEN_WIDTH)
    return ZooRunIdentity(
        task_id=stream.task_id,
        suite="libero_spatial",
        task_index=0,
        instruction=stream.instruction,
        seed=seed,
        manifest_name="tiny",
        zoo_sha256="z" * 64,
        budget=budget.training_identity(),
        snapshot_updates=tuple(budget.snapshot_updates),
        purpose="zoo",
        dataset_sha256=dataset_sha256,
        train_episode_ids=tuple(item.episode_id for item in stream.episodes),
        held_out_episode_ids=tuple(item.episode_id for item in held_out.episodes),
        base_sha256=bundle.base_sha256,
        base_revision=bundle.base_revision,
        architecture_sha256=bundle.architecture_sha256,
        preprocessing_sha256=bundle.preprocessing.sha256(),
        action_convention_sha256=bundle.action_convention.sha256(),
        adapter_spec_sha256=spec.sha256(),
        environment_profile="test",
    )
