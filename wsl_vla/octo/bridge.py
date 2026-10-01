"""Official Octo integration with modality-masked per-block residual adapters."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from ..experiments.provenance import sha256_array_tree


OCTO_GIT_REVISION = "241fb3514b7c40957a86d869fecb7c7fc353f540"
OCTO_MODEL_ID = "rail-berkeley/octo-small-1.5"
OCTO_MODEL_REVISION = "dc9aa3019f764726c770814b27e4ab0fc6e32a58"


def _research_imports():
    try:
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
        import octo.model.octo_module as octo_module
        from octo.model.components.block_transformer import BlockTransformer
        from octo.model.components.transformer import Encoder1DBlock
        from octo.model.octo_model import OctoModel
        from octo.utils.train_utils import merge_params
    except ImportError as exc:
        raise RuntimeError(
            "Official Octo integration requires the pinned Linux/WSL2 research environment. "
            "Install requirements-research.txt under Python 3.10 or 3.11."
        ) from exc
    return nn, jax, jnp, octo_module, BlockTransformer, Encoder1DBlock, OctoModel, merge_params


def _group_component(name: str) -> int:
    if name.startswith("task_"):
        return 1  # language
    # Octo can repeat task tokens at each observation timestep and renames those
    # groups ``obs_task_*``. They remain language evidence and must not be
    # routed through the visual adapter merely because of the synthetic prefix.
    if name.startswith("obs_task_"):
        return 1  # language
    if name.startswith("obs_"):
        return 0  # vision
    if name.startswith("readout_"):
        return 2  # action
    raise ValueError(f"unrecognized Octo token group: {name}")


def install_octo_modality_patch(*, rank: int, alpha: float):
    """Patch Octo's block transformer before creating a research model.

    The pretrained parameter names are unchanged. New adapter ``up`` kernels are
    initialized to zero, so the patched model is functionally identical before
    adaptation. The returned class is also assigned to Octo's module namespace.
    """

    if rank <= 0 or alpha <= 0:
        raise ValueError("rank and alpha must be positive")
    nn, _, jnp, octo_module, BlockTransformer, Encoder1DBlock, _, _ = _research_imports()
    from octo.model.components.transformer import AddPositionEmbs

    class ResearchTransformer(nn.Module):
        """Official Transformer parameter hierarchy plus modality adapters."""

        transformer_kwargs: dict
        adapter_rank: int
        adapter_alpha: float

        @nn.compact
        def __call__(self, x, attention_mask, component_ids, *, train: bool):
            kwargs = self.transformer_kwargs
            if kwargs.get("add_position_embedding", False):
                x = AddPositionEmbs(
                    posemb_init=nn.initializers.normal(stddev=0.02),
                    name="posembed_input",
                )(x)
                x = nn.Dropout(rate=kwargs.get("dropout_rate", 0.1))(
                    x, deterministic=not train
                )
            for layer_index in range(kwargs["num_layers"]):
                x = Encoder1DBlock(
                    mlp_dim=kwargs["mlp_dim"],
                    dropout_rate=kwargs.get("dropout_rate", 0.1),
                    attention_dropout_rate=kwargs.get("attention_dropout_rate", 0.1),
                    name=f"encoderblock_{layer_index}",
                    num_heads=kwargs["num_attention_heads"],
                )(x, attention_mask, deterministic=not train)
                for component_index, component_name in enumerate(
                    ("vision", "language", "action")
                ):
                    mask = (component_ids == component_index)[None, :, None]
                    down = nn.Dense(
                        self.adapter_rank,
                        use_bias=False,
                        kernel_init=nn.initializers.normal(stddev=0.02),
                        name=f"adapter_{component_name}_{layer_index}_down",
                    )(x)
                    delta = nn.Dense(
                        x.shape[-1],
                        use_bias=False,
                        kernel_init=nn.initializers.zeros,
                        name=f"adapter_{component_name}_{layer_index}_up",
                    )(down)
                    decoded_kernel = self.param(
                        f"decoded_{component_name}_{layer_index}_kernel",
                        nn.initializers.zeros,
                        (x.shape[-1], x.shape[-1]),
                    )
                    decoded_delta = jnp.einsum("...d,df->...f", x, decoded_kernel)
                    x = x + mask * (
                        delta * (self.adapter_alpha / self.adapter_rank) + decoded_delta
                    )
            return nn.LayerNorm(name="encoder_norm")(x)

    class ResearchBlockTransformer(BlockTransformer):
        adapter_rank: int = rank
        adapter_alpha: float = alpha

        @nn.compact
        def __call__(self, prefix_groups, timestep_groups, train: bool, verbose: bool = False):
            if verbose:
                self.pretty_print_attention_mask(prefix_groups, timestep_groups)
            horizon = timestep_groups[0].tokens.shape[1]
            if not all(group.tokens.shape[1] == horizon for group in timestep_groups):
                raise ValueError("Octo timestep token groups have inconsistent horizons")
            input_tokens = self.assemble_input_tokens(prefix_groups, timestep_groups)
            attention_mask = self.generate_attention_mask(prefix_groups, timestep_groups)
            self.sow("intermediates", "attention_mask", attention_mask)

            prefix_ids = [
                jnp.full((group.tokens.shape[1],), _group_component(group.name), dtype=jnp.int32)
                for group in prefix_groups
            ]
            step_ids = [
                jnp.full((group.tokens.shape[2],), _group_component(group.name), dtype=jnp.int32)
                for group in timestep_groups
            ]
            prefix_vector = (
                jnp.concatenate(prefix_ids)
                if prefix_ids
                else jnp.zeros((0,), dtype=jnp.int32)
            )
            timestep_vector = jnp.tile(jnp.concatenate(step_ids), horizon)
            component_ids = jnp.concatenate([prefix_vector, timestep_vector])
            output = ResearchTransformer(
                transformer_kwargs=self.transformer_kwargs,
                adapter_rank=self.adapter_rank,
                adapter_alpha=self.adapter_alpha,
                name="Transformer_0",
            )(input_tokens, attention_mask, component_ids, train=train)
            return self.split_output_tokens(output, prefix_groups, timestep_groups)

    ResearchBlockTransformer.__name__ = "BlockTransformer"
    octo_module.BlockTransformer = ResearchBlockTransformer
    return ResearchBlockTransformer


@dataclass(frozen=True)
class ResearchOctoBundle:
    pretrained_model: Any
    research_model: Any
    base_sha256: str
    base_revision: str
    adapter_rank: int
    adapter_alpha: float
    diffusion_kernel_paths: tuple[tuple[str, ...], ...]
    diffusion_factors: Any
    reference_transformer_outputs: Any


def load_research_octo(
    checkpoint: str = f"hf://{OCTO_MODEL_ID}",
    *,
    step: int | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    seed: int = 0,
) -> ResearchOctoBundle:
    """Load official weights, create the patched module, and merge by key/shape."""

    _, jax, _, octo_module, BlockTransformer, _, OctoModel, merge_params = _research_imports()
    # ``install_octo_modality_patch`` mutates Octo's module global. Restore the
    # official class first so repeated loads in one process cannot accidentally
    # treat an already-patched graph as the pretrained reference.
    octo_module.BlockTransformer = BlockTransformer
    resolved_checkpoint = checkpoint
    resolved_revision = "local-unresolved"
    if checkpoint.startswith("hf://"):
        repository = checkpoint.removeprefix("hf://").rstrip("/")
        if repository != OCTO_MODEL_ID:
            raise ValueError(f"research backbone must be {OCTO_MODEL_ID}, got {repository}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for pinned checkpoint loading") from exc
        resolved_checkpoint = snapshot_download(
            repo_id=repository,
            revision=OCTO_MODEL_REVISION,
        )
        resolved_revision = OCTO_MODEL_REVISION
    pretrained = OctoModel.load_pretrained(resolved_checkpoint, step=step)
    base_sha256 = sha256_array_tree(pretrained.params)
    if resolved_revision == "local-unresolved":
        resolved_revision = f"local-sha256:{base_sha256}"
    reference_observations = pretrained.example_batch["observation"]
    reference_tasks = pretrained.example_batch["task"]
    reference_timestep_mask = reference_observations["timestep_pad_mask"]
    # Materialize the true official output before changing Octo's module global.
    # ``device_get`` also closes the door on asynchronous dispatch observing the
    # later monkeypatch.
    reference_transformer_outputs = jax.device_get(
        pretrained.run_transformer(
            reference_observations,
            reference_tasks,
            reference_timestep_mask,
            train=False,
        )
    )

    install_octo_modality_patch(rank=rank, alpha=alpha)
    research = OctoModel.from_config(
        pretrained.config,
        pretrained.example_batch,
        pretrained.text_processor,
        rng=jax.random.PRNGKey(seed),
        dataset_statistics=pretrained.dataset_statistics,
    )
    merged = merge_params(research.params, pretrained.params)
    research = research.replace(params=merged)
    from ..adapters.flax import (
        discover_diffusion_kernel_paths,
        initialize_external_adapters,
    )

    diffusion_paths = discover_diffusion_kernel_paths(research.params)
    diffusion_factors = initialize_external_adapters(
        research.params, diffusion_paths, rank=rank, seed=seed + 1
    )
    return ResearchOctoBundle(
        pretrained,
        research,
        base_sha256,
        resolved_revision,
        rank,
        alpha,
        diffusion_paths,
        diffusion_factors,
        reference_transformer_outputs,
    )


def assert_zero_adapter_equivalence(bundle: ResearchOctoBundle, *, atol: float = 1e-6) -> None:
    """Verify that adding zero adapters leaves every transformer output unchanged."""

    _, jax, jnp, _, _, _, _, _ = _research_imports()
    from ..adapters.flax import apply_external_adapters

    observations = bundle.pretrained_model.example_batch["observation"]
    tasks = bundle.pretrained_model.example_batch["task"]
    timestep_mask = observations["timestep_pad_mask"]
    # This was computed before Octo's module-global BlockTransformer was
    # patched. Re-running the pretrained module here would no longer be a valid
    # unmodified reference.
    original = bundle.reference_transformer_outputs
    parameters = apply_external_adapters(
        bundle.research_model.params,
        bundle.diffusion_factors,
        alpha=bundle.adapter_alpha,
    )
    patched_model = bundle.research_model.replace(params=parameters)
    patched = patched_model.run_transformer(observations, tasks, timestep_mask, train=False)
    original_head = bundle.pretrained_model.module.bind(
        {"params": bundle.pretrained_model.params}
    ).heads["action"]
    patched_head = patched_model.module.bind({"params": parameters}).heads["action"]
    time = jnp.zeros((*timestep_mask.shape, 1), dtype=jnp.float32)
    noisy_actions = jnp.zeros(
        (*timestep_mask.shape, original_head.action_horizon * original_head.action_dim),
        dtype=jnp.float32,
    )
    original_prediction = original_head(
        original, time=time, noisy_actions=noisy_actions, train=False
    )
    patched_prediction = patched_head(
        patched, time=time, noisy_actions=noisy_actions, train=False
    )
    original_leaves = jax.tree_util.tree_leaves((original, original_prediction))
    patched_leaves = jax.tree_util.tree_leaves((patched, patched_prediction))
    if len(original_leaves) != len(patched_leaves):
        raise AssertionError("patched Octo output tree differs from the official model")
    for index, (left, right) in enumerate(zip(original_leaves, patched_leaves)):
        if np.asarray(left).dtype.kind not in "biufc":
            continue
        if not np.allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=0):
            difference = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
            raise AssertionError(
                f"zero-adapter equivalence failed at output leaf {index}; max abs diff={difference}"
            )


def adapter_parameter_paths(params: Any) -> tuple[tuple[str, ...], ...]:
    """Return all patched adapter leaf paths in deterministic order."""

    try:
        import flax.traverse_util
    except ImportError as exc:
        raise RuntimeError("Flax is required to inspect Octo adapter parameters") from exc
    flat = flax.traverse_util.flatten_dict(params)
    return tuple(sorted(path for path in flat if any(str(part).startswith("adapter_") for part in path)))


def build_adapter_spec_and_factors(
    bundle: ResearchOctoBundle,
    *,
    token_width: int = 384,
    adapter_state: Any | None = None,
):
    """Extract every transformer and diffusion-head factor in canonical form."""

    try:
        import flax.traverse_util
        from flax.core import unfreeze
    except ImportError as exc:
        raise RuntimeError("Flax is required to extract Octo adapters") from exc
    from ..adapters.flax import path_string
    from ..contracts import AdapterEntry, AdapterSpec, Component

    flat = flax.traverse_util.flatten_dict(unfreeze(bundle.research_model.params))
    entries_and_factors = []
    pattern = re.compile(r"adapter_(vision|language|action)_(\d+)_(down|up)$")
    for path, value in flat.items():
        if path[-1] != "kernel":
            continue
        matched_index = None
        match = None
        for index, part in enumerate(path):
            candidate = pattern.fullmatch(str(part))
            if candidate and candidate.group(3) == "down":
                matched_index, match = index, candidate
                break
        if match is None:
            continue
        up_path = list(path)
        up_path[matched_index] = str(path[matched_index]).removesuffix("_down") + "_up"
        up_path = tuple(up_path)
        if up_path not in flat:
            raise ValueError(f"missing paired up factor for {path_string(path)}")
        if adapter_state is None:
            down, up = value, flat[up_path]
        else:
            down_key = path_string(path)
            up_key = path_string(up_path)
            try:
                down = adapter_state["transformer"][down_key]
                up = adapter_state["transformer"][up_key]
            except KeyError as exc:
                raise ValueError(f"trained adapter state lacks {exc.args[0]}") from exc
        logical_path = path_string(path[:matched_index] + (str(path[matched_index]).removesuffix("_down"),))
        entry = AdapterEntry(
            component=Component(match.group(1)),
            layer=int(match.group(2)),
            parameter_path=logical_path,
            input_dim=int(down.shape[0]),
            output_dim=int(up.shape[1]),
            rank=int(down.shape[1]),
        )
        entries_and_factors.append((entry, (down, up)))

    for head_index, path in enumerate(bundle.diffusion_kernel_paths):
        encoded = path_string(path)
        pair = (
            bundle.diffusion_factors[encoded]
            if adapter_state is None
            else adapter_state["diffusion"][encoded]
        )
        down, up = pair["down"], pair["up"]
        entry = AdapterEntry(
            component=Component.ACTION,
            layer=head_index,
            parameter_path=encoded,
            input_dim=int(down.shape[0]),
            output_dim=int(up.shape[1]),
            rank=int(down.shape[1]),
        )
        entries_and_factors.append((entry, (down, up)))

    if not entries_and_factors:
        raise ValueError("no research adapters found in the Octo parameter tree")
    entries_and_factors.sort(
        key=lambda item: (item[0].component.value, item[0].layer, item[0].parameter_path)
    )
    spec = AdapterSpec(
        base_model_id=OCTO_MODEL_ID,
        base_revision=bundle.base_revision,
        base_sha256=bundle.base_sha256,
        alpha=bundle.adapter_alpha,
        token_width=token_width,
        entries=tuple(item[0] for item in entries_and_factors),
    )
    return spec, {entry.parameter_path: pair for entry, pair in entries_and_factors}
