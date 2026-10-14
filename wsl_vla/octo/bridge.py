"""Official Octo integration with token-conditioned, pre-block modality adapters.

Octo adaptation of the proposal
-------------------------------
The proposal factorizes a VLA update into vision, language, and action
components and assumes a separate set of weights per component. Octo-Small has
one shared transformer over a mixed token sequence, so the factorization here is
*token-conditioned*: each transformer block owns three zero-initialized rank-r
residuals, and a token only ever receives the residual of its own modality.

* image tokens (``obs_primary``, ``obs_wrist``) -> vision adapter;
* language tokens (``task_language`` and its per-timestep repeat
  ``obs_task_language``) -> language adapter;
* action readout tokens -> action adapter, together with low-rank updates of
  every 2-D kernel in the diffusion action head.

Placement
---------
Residuals are added to the *input* of each block, and all three are computed
from that same block input. The earlier post-block placement left the
final-layer vision and language residuals with no path to the action readout:
after the last block only a token-wise LayerNorm remains, so those adapters got
exactly zero gradient and stayed zero in every zoo sample.

Routing is an explicit table derived from the checkpoint configuration. A token
group that is neither routed nor explicitly frozen is an error; nothing is
labelled "vision" by default.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

import numpy as np

from ..contracts import (
    COMPONENT_NAMES,
    FROZEN_ROUTE,
    LIBERO_ACTION_CONVENTION,
    SITE_BLOCK_INPUT,
    SITE_HEAD_KERNEL,
    ActionConvention,
)
from ..data.actions import assert_head_preserves_targets
from ..data.preprocessing import PreprocessingSpec, preprocessing_spec_from_checkpoint
from ..experiments.provenance import sha256_array_tree
from .compat import (
    OCTO_PATCHED_REVISION,
    architecture_sha256,
    true_float32_matmul,
)


# Blackwell profile: upstream Octo plus third_party/octo/*.patch (JAX 0.7 port),
# rebuilt as a deterministic commit by scripts/setup/build_patched_octo.sh. The
# legacy-gpu profile installs the unmodified upstream revision.
OCTO_GIT_REVISION = OCTO_PATCHED_REVISION
OCTO_MODEL_ID = "rail-berkeley/octo-small-1.5"
OCTO_MODEL_REVISION = "dc9aa3019f764726c770814b27e4ab0fc6e32a58"
EQUIVALENCE_SEED = 20261014
FROZEN_COMPONENT_ID = -1
ADAPTER_MARKER_KEY = "research_adapter"
ADAPTER_LEAF = re.compile(r"adapter_(vision|language|action)_(\d+)_(down|up)$")
DECODED_LEAF = re.compile(r"decoded_(vision|language|action)_(\d+)_kernel$")


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
            "Official Octo integration requires one of the locked research environments. "
            "Run envs/scripts/setup_research.sh --profile <blackwell|legacy-gpu>."
        ) from exc
    return nn, jax, jnp, octo_module, BlockTransformer, Encoder1DBlock, OctoModel, merge_params


def _tokenizer_route(kind: str, name: str, spec: Mapping[str, Any]) -> str:
    class_name = spec.get("name")
    if class_name == "ImageTokenizer":
        return "vision"
    if class_name == "LanguageTokenizer":
        return "language"
    if class_name == "LowdimObsTokenizer":
        # Proprioception and other low-dimensional observations have no modality
        # adapter in this study; their tokens pass through every block unchanged.
        return FROZEN_ROUTE
    raise ValueError(
        f"unrecognized Octo {kind} tokenizer {name!r} of type {class_name!r}: "
        "add an explicit routing rule instead of guessing its modality"
    )


def build_token_routing(model_config: Mapping[str, Any]) -> dict[str, str]:
    """Explicit ``token group -> modality | frozen`` table for one checkpoint."""

    routing: dict[str, str] = {}

    def assign(group: str, route: str) -> None:
        if group in routing:
            raise ValueError(f"Octo token group {group!r} is produced twice")
        routing[group] = route

    for name, spec in model_config.get("task_tokenizers", {}).items():
        route = _tokenizer_route("task", name, spec)
        assign(f"task_{name}", route)
        if model_config.get("repeat_task_tokens", False):
            # Octo re-inserts task tokens at every timestep under an ``obs_``
            # prefix; they are still language (or goal-image) tokens.
            assign(f"obs_task_{name}", route)
    for name, spec in model_config.get("observation_tokenizers", {}).items():
        assign(f"obs_{name}", _tokenizer_route("observation", name, spec))
    action_readouts = {
        str(head["kwargs"]["readout_key"])
        for head in model_config.get("heads", {}).values()
        if "readout_key" in head.get("kwargs", {})
    }
    for name in model_config.get("readouts", {}):
        group = f"readout_{name}"
        assign(group, "action" if group in action_readouts else FROZEN_ROUTE)
    missing = [name for name in COMPONENT_NAMES if name not in routing.values()]
    if missing:
        raise ValueError(f"checkpoint routes no token group to: {missing}")
    return dict(sorted(routing.items()))


def adapter_marker(*, rank: int, alpha: float, routing: Mapping[str, str]) -> str:
    """Canonical identity of one adapter layout: rank, scale, and token routing."""

    return json.dumps(
        {"rank": int(rank), "alpha": float(alpha), "routing": dict(sorted(routing.items()))},
        sort_keys=True,
        separators=(",", ":"),
    )


def research_model_config(
    config: Mapping[str, Any], *, rank: int, alpha: float, routing: Mapping[str, str]
) -> dict[str, Any]:
    """Copy of an official Octo config whose transformer names its adapter layout.

    Octo instantiates its block transformer from a module global, so nothing in
    an official module records which patched class it was built with. Writing
    the adapter identity into ``transformer_kwargs`` makes it part of the Flax
    module: compiled functions of two research models can never be confused,
    and a module traced while a different patch is installed fails loudly.
    """

    research = copy.deepcopy(dict(config))
    model = dict(research["model"])
    model["transformer_kwargs"] = {
        **dict(model["transformer_kwargs"]),
        ADAPTER_MARKER_KEY: adapter_marker(rank=rank, alpha=alpha, routing=routing),
    }
    research["model"] = model
    return research


def install_octo_modality_patch(*, rank: int, alpha: float, routing: Mapping[str, str]):
    """Patch Octo's block transformer before creating a research model.

    The pretrained parameter names are unchanged. New adapter ``up`` kernels are
    initialized to zero, so the patched model is functionally identical before
    adaptation. The returned class is also assigned to Octo's module namespace;
    it only accepts modules created from :func:`research_model_config` with the
    same rank, scale, and routing.
    """

    if rank <= 0 or alpha <= 0:
        raise ValueError("rank and alpha must be positive")
    unknown = sorted(set(routing.values()) - {*COMPONENT_NAMES, FROZEN_ROUTE})
    if unknown:
        raise ValueError(f"token routing uses unknown destinations: {unknown}")
    identity = adapter_marker(rank=rank, alpha=alpha, routing=routing)
    route_ids = {
        group: FROZEN_COMPONENT_ID if route == FROZEN_ROUTE else COMPONENT_NAMES.index(route)
        for group, route in routing.items()
    }
    nn, _, jnp, octo_module, BlockTransformer, Encoder1DBlock, _, _ = _research_imports()
    from octo.model.components.transformer import AddPositionEmbs

    def group_component(name: str) -> int:
        if name not in route_ids:
            raise ValueError(
                f"unrecognized Octo token group {name!r}; routed groups are "
                f"{sorted(route_ids)}. Refusing to assign it a modality."
            )
        return route_ids[name]

    class ResearchTransformer(nn.Module):
        """Official Transformer parameter hierarchy plus pre-block modality adapters."""

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
            scale = self.adapter_alpha / self.adapter_rank
            for layer_index in range(kwargs["num_layers"]):
                # Every modality residual reads the same block input, so the
                # three branches are independent of evaluation order.
                block_input = x
                residual = jnp.zeros_like(block_input)
                for component_index, component_name in enumerate(COMPONENT_NAMES):
                    mask = (component_ids == component_index)[None, :, None]
                    down = nn.Dense(
                        self.adapter_rank,
                        use_bias=False,
                        kernel_init=nn.initializers.normal(stddev=0.02),
                        name=f"adapter_{component_name}_{layer_index}_down",
                    )(block_input)
                    delta = nn.Dense(
                        block_input.shape[-1],
                        use_bias=False,
                        kernel_init=nn.initializers.zeros,
                        name=f"adapter_{component_name}_{layer_index}_up",
                    )(down)
                    decoded_kernel = self.param(
                        f"decoded_{component_name}_{layer_index}_kernel",
                        nn.initializers.zeros,
                        (block_input.shape[-1], block_input.shape[-1]),
                    )
                    decoded_delta = jnp.einsum("...d,df->...f", block_input, decoded_kernel)
                    residual = residual + jnp.where(mask, delta * scale + decoded_delta, 0.0)
                # Recorded only when a caller asks for intermediates (routing audit).
                self.sow("intermediates", f"adapter_residual_{layer_index}", residual)
                x = Encoder1DBlock(
                    mlp_dim=kwargs["mlp_dim"],
                    dropout_rate=kwargs.get("dropout_rate", 0.1),
                    attention_dropout_rate=kwargs.get("attention_dropout_rate", 0.1),
                    name=f"encoderblock_{layer_index}",
                    num_heads=kwargs["num_attention_heads"],
                )(block_input + residual, attention_mask, deterministic=not train)
            return nn.LayerNorm(name="encoder_norm")(x)

    class ResearchBlockTransformer(BlockTransformer):
        adapter_rank: int = rank
        adapter_alpha: float = alpha

        @nn.compact
        def __call__(self, prefix_groups, timestep_groups, train: bool, verbose: bool = False):
            if self.transformer_kwargs.get(ADAPTER_MARKER_KEY) != identity:
                raise RuntimeError(
                    "this Octo module was not built for the research transformer that is "
                    "currently installed (different rank, alpha, routing, or an official "
                    "model). Call activate_research_bundle(bundle) before tracing a research "
                    "model and restore_official_octo() before tracing a pretrained one."
                )
            if verbose:
                self.pretty_print_attention_mask(prefix_groups, timestep_groups)
            horizon = timestep_groups[0].tokens.shape[1]
            if not all(group.tokens.shape[1] == horizon for group in timestep_groups):
                raise ValueError("Octo timestep token groups have inconsistent horizons")
            input_tokens = self.assemble_input_tokens(prefix_groups, timestep_groups)
            attention_mask = self.generate_attention_mask(prefix_groups, timestep_groups)
            self.sow("intermediates", "attention_mask", attention_mask)

            prefix_ids = [
                np.full((group.tokens.shape[1],), group_component(group.name), dtype=np.int32)
                for group in prefix_groups
            ]
            step_ids = [
                np.full((group.tokens.shape[2],), group_component(group.name), dtype=np.int32)
                for group in timestep_groups
            ]
            prefix_vector = (
                np.concatenate(prefix_ids) if prefix_ids else np.zeros((0,), dtype=np.int32)
            )
            timestep_vector = np.tile(np.concatenate(step_ids), horizon)
            component_ids = jnp.asarray(np.concatenate([prefix_vector, timestep_vector]))
            self.sow("intermediates", "adapter_component_ids", component_ids)
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
    architecture_sha256: str
    adapter_rank: int
    adapter_alpha: float
    token_routing: Mapping[str, str]
    action_convention: ActionConvention
    preprocessing: PreprocessingSpec
    feature_extractor_sha256: Mapping[str, str]
    diffusion_kernel_paths: tuple[tuple[str, ...], ...]
    diffusion_factors: Any
    reference_transformer_outputs: Any
    reference_sampled_actions: Any
    equivalence_seed: int = EQUIVALENCE_SEED
    patched_block_transformer: Any = None

    def identity(self) -> dict[str, Any]:
        """Independently recorded checkpoint, architecture, preprocessing, and controller hashes."""

        return {
            "base_model_revision": self.base_revision,
            "base_sha256": self.base_sha256,
            "architecture_sha256": self.architecture_sha256,
            "preprocessing_sha256": self.preprocessing.sha256(),
            "action_convention_sha256": self.action_convention.sha256(),
            "action_convention": self.action_convention.name,
            "adapter_rank": int(self.adapter_rank),
            "adapter_alpha": float(self.adapter_alpha),
            "token_routing": dict(self.token_routing),
            "consumed_observations": list(self.preprocessing.consumed_observations),
            "proprio_consumed": bool(self.preprocessing.proprio_consumed),
            "feature_extractor_sha256": dict(self.feature_extractor_sha256),
        }


def is_adapter_path(path: tuple[Any, ...]) -> bool:
    return any(ADAPTER_LEAF.fullmatch(str(part)) for part in path)


def is_decoded_path(path: tuple[Any, ...]) -> bool:
    return any(DECODED_LEAF.fullmatch(str(part)) for part in path)


def base_parameter_sha256(params: Any) -> str:
    """Hash a research parameter tree with every adapter leaf removed."""

    try:
        import flax.traverse_util
        from flax.core import unfreeze
    except ImportError as exc:
        raise RuntimeError("Flax is required to inspect Octo parameters") from exc
    flat = flax.traverse_util.flatten_dict(unfreeze(params))
    base = {
        path: value
        for path, value in flat.items()
        if not is_adapter_path(path) and not is_decoded_path(path)
    }
    return sha256_array_tree(flax.traverse_util.unflatten_dict(base))


def unnormalization_statistics(convention: ActionConvention) -> dict[str, np.ndarray]:
    """Identity statistics: Octo samples directly in controller coordinates."""

    return {
        "mean": np.zeros(convention.action_dim, dtype=np.float32),
        "std": np.ones(convention.action_dim, dtype=np.float32),
    }


def _feature_extractor_hashes(pretrained: Any, architecture: str) -> dict[str, str]:
    transformer = pretrained.params["octo_transformer"]
    config = pretrained.config["model"]
    hashes = {}
    for label, family, class_name in (
        ("vision", "observation_tokenizers", "ImageTokenizer"),
        ("language", "task_tokenizers", "LanguageTokenizer"),
    ):
        names = sorted(
            name for name, spec in config[family].items() if spec.get("name") == class_name
        )
        subtree = {
            f"{family}_{name}": transformer[f"{family}_{name}"]
            for name in names
            if f"{family}_{name}" in transformer
        }
        if not subtree:
            raise ValueError(f"checkpoint has no frozen {label} feature extractor parameters")
        import hashlib

        hashes[label] = hashlib.sha256(
            (architecture + sha256_array_tree(subtree)).encode("utf-8")
        ).hexdigest()
    return hashes


def build_research_bundle(
    pretrained: Any,
    *,
    base_revision: str,
    rank: int,
    alpha: float,
    seed: int,
    preprocessing: Mapping[str, Any],
    action_convention: ActionConvention = LIBERO_ACTION_CONVENTION,
    equivalence_seed: int = EQUIVALENCE_SEED,
) -> ResearchOctoBundle:
    """Wrap an already-loaded pretrained ``OctoModel`` with research adapters.

    ``preprocessing`` is the configuration section naming the source camera
    resolution, orientation handling, and executed actions per plan; nothing
    about model inputs is defaulted here.
    """

    _, jax, _, _, _, _, OctoModel, merge_params = _research_imports()
    # ``install_octo_modality_patch`` mutates Octo's module global. Restore the
    # official class first so repeated loads in one process cannot accidentally
    # treat an already-patched graph as the pretrained reference.
    restore_official_octo()
    model_config = pretrained.config["model"]
    routing = build_token_routing(model_config)
    architecture = architecture_sha256(pretrained)
    spec = preprocessing_spec_from_checkpoint(
        model_config=model_config,
        example_batch=pretrained.example_batch,
        text_processor_config=pretrained.config.get("text_processor"),
        text_processor=pretrained.text_processor,
        source_resolution=(
            int(preprocessing["source_camera_height"]),
            int(preprocessing["source_camera_width"]),
        ),
        source_image_convention=str(preprocessing["demonstration_image_convention"]),
        image_transform=str(preprocessing["image_transform"]),
        simulator_alignment=str(preprocessing["simulator_alignment"]),
        executed_actions_per_plan=int(preprocessing["executed_actions_per_plan"]),
    )
    if spec.action_dim != action_convention.action_dim:
        raise ValueError(
            f"checkpoint predicts {spec.action_dim}-D actions but the controller "
            f"convention {action_convention.name} has {action_convention.action_dim}"
        )
    head = pretrained.module.heads["action"]
    max_action = getattr(head, "max_action", None)
    if max_action is not None:
        assert_head_preserves_targets(action_convention, head_max_action=float(max_action))

    base_sha256 = sha256_array_tree(pretrained.params)
    reference_observations = pretrained.example_batch["observation"]
    reference_tasks = pretrained.example_batch["task"]
    reference_timestep_mask = reference_observations["timestep_pad_mask"]
    # Materialize the true official outputs before changing Octo's module
    # global; ``device_get`` also closes the door on asynchronous dispatch
    # observing the later monkeypatch.
    with true_float32_matmul():
        reference_transformer_outputs = jax.device_get(
            pretrained.run_transformer(
                reference_observations, reference_tasks, reference_timestep_mask, train=False
            )
        )
        reference_sampled_actions = jax.device_get(
            pretrained.sample_actions(
                reference_observations,
                reference_tasks,
                unnormalization_statistics=unnormalization_statistics(action_convention),
                rng=jax.random.PRNGKey(equivalence_seed),
            )
        )

    patched = install_octo_modality_patch(rank=rank, alpha=alpha, routing=routing)
    research = OctoModel.from_config(
        research_model_config(pretrained.config, rank=rank, alpha=alpha, routing=routing),
        pretrained.example_batch,
        pretrained.text_processor,
        rng=jax.random.PRNGKey(seed),
        dataset_statistics=pretrained.dataset_statistics,
    )
    merged = merge_params(research.params, pretrained.params)
    if base_parameter_sha256(merged) != base_sha256:
        raise RuntimeError(
            "patched Octo did not inherit every pretrained parameter; the research "
            "model would not share the pinned base"
        )
    # One device copy of the frozen base, shared by every compiled function and
    # every materialized policy (they receive it as an argument, never as an
    # embedded constant, and never re-upload host arrays per call).
    research = research.replace(params=jax.device_put(merged))
    from ..adapters.flax import discover_diffusion_kernel_paths, initialize_external_adapters

    diffusion_paths = discover_diffusion_kernel_paths(research.params)
    diffusion_factors = initialize_external_adapters(
        research.params, diffusion_paths, rank=rank, seed=seed + 1
    )
    return ResearchOctoBundle(
        pretrained_model=pretrained,
        research_model=research,
        base_sha256=base_sha256,
        base_revision=base_revision,
        architecture_sha256=architecture,
        adapter_rank=rank,
        adapter_alpha=alpha,
        token_routing=routing,
        action_convention=action_convention,
        preprocessing=spec,
        feature_extractor_sha256=_feature_extractor_hashes(pretrained, architecture),
        diffusion_kernel_paths=diffusion_paths,
        diffusion_factors=diffusion_factors,
        reference_transformer_outputs=reference_transformer_outputs,
        reference_sampled_actions=reference_sampled_actions,
        equivalence_seed=equivalence_seed,
        patched_block_transformer=patched,
    )


def restore_official_octo() -> None:
    """Point Octo back at its unmodified block transformer.

    Required before a pretrained checkpoint is loaded or traced: Octo builds
    its parameter template from whatever class the module global names, so a
    patch left behind by an earlier bundle would be mistaken for the base.
    """

    _, _, _, octo_module, BlockTransformer, _, _, _ = _research_imports()
    octo_module.BlockTransformer = BlockTransformer


def activate_research_bundle(bundle: ResearchOctoBundle) -> None:
    """Make ``bundle``'s adapter graph the one Octo instantiates from now on.

    The modality patch lives in a module global that Octo reads every time its
    transformer is traced. Loading a second bundle (another rank, routing, or
    checkpoint) replaces that global, so each bundle re-installs its own patched
    class before any of its graphs is traced. Without this, a later trace of the
    first bundle would silently build the second bundle's adapter layout.
    """

    if bundle.patched_block_transformer is None:
        raise RuntimeError("research bundle carries no patched Octo transformer")
    *_, octo_module, _, _, _, _ = _research_imports()
    octo_module.BlockTransformer = bundle.patched_block_transformer


def trace_adapter_routing(
    bundle: ResearchOctoBundle, adapter_state: Any | None = None
) -> dict[str, Any]:
    """Audit the routing the live patched graph applies to the example batch.

    Returns the component id of every token position (``-1`` marks a frozen
    group), the token count per route, and the per-layer residual that was
    actually added. This reads the graph rather than the routing table, so it
    catches a table that is correct on paper but mis-applied in the module.
    """

    _, jax, _, _, _, _, _, _ = _research_imports()
    import flax.traverse_util
    from flax.core import unfreeze

    from .training import initial_adapter_state, materialize_policy_params

    state = initial_adapter_state(bundle) if adapter_state is None else adapter_state
    params = materialize_policy_params(bundle, state)
    observations = bundle.pretrained_model.example_batch["observation"]
    tasks = bundle.pretrained_model.example_batch["task"]
    _, captured = bundle.research_model.module.apply(
        {"params": params},
        observations,
        tasks,
        observations["timestep_pad_mask"],
        train=False,
        method="octo_transformer",
        mutable=["intermediates"],
    )
    flat = flax.traverse_util.flatten_dict(unfreeze(captured["intermediates"]))
    by_name = {str(path[-1]): values for path, values in flat.items()}
    if "adapter_component_ids" not in by_name:
        raise RuntimeError("the live Octo graph is not the patched research transformer")
    component_ids = np.asarray(jax.device_get(by_name["adapter_component_ids"][0]))
    residuals = {
        int(name.rsplit("_", 1)[1]): np.asarray(jax.device_get(values[0]))
        for name, values in by_name.items()
        if name.startswith("adapter_residual_")
    }
    counts = {
        name: int(np.sum(component_ids == index)) for index, name in enumerate(COMPONENT_NAMES)
    }
    counts[FROZEN_ROUTE] = int(np.sum(component_ids == FROZEN_COMPONENT_ID))
    if sum(counts.values()) != component_ids.size:
        raise RuntimeError("the live Octo graph routes tokens to an unknown destination")
    return {
        "component_ids": component_ids,
        "token_counts": counts,
        "residuals": dict(sorted(residuals.items())),
    }


def load_research_octo(
    checkpoint: str = f"hf://{OCTO_MODEL_ID}",
    *,
    step: int | None = None,
    rank: int,
    alpha: float,
    seed: int,
    preprocessing: Mapping[str, Any],
) -> ResearchOctoBundle:
    """Load official weights, create the patched module, and merge by key/shape."""

    *_, OctoModel, _ = _research_imports()
    resolved_checkpoint = checkpoint
    resolved_revision = None
    if checkpoint.startswith("hf://"):
        repository = checkpoint.removeprefix("hf://").rstrip("/")
        if repository != OCTO_MODEL_ID:
            raise ValueError(f"research backbone must be {OCTO_MODEL_ID}, got {repository}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for pinned checkpoint loading") from exc
        resolved_checkpoint = snapshot_download(repo_id=repository, revision=OCTO_MODEL_REVISION)
        resolved_revision = OCTO_MODEL_REVISION
    restore_official_octo()
    pretrained = OctoModel.load_pretrained(resolved_checkpoint, step=step)
    if resolved_revision is None:
        resolved_revision = f"local-sha256:{sha256_array_tree(pretrained.params)}"
    return build_research_bundle(
        pretrained,
        base_revision=resolved_revision,
        rank=rank,
        alpha=alpha,
        seed=seed,
        preprocessing=preprocessing,
    )


def load_research_octo_from_config(
    config: Mapping[str, Any], checkpoint: str, *, seed: int
) -> ResearchOctoBundle:
    """Resolve adapter and preprocessing settings from the research configuration."""

    return load_research_octo(
        checkpoint,
        rank=int(config["adapter"]["rank"]),
        alpha=float(config["adapter"]["alpha"]),
        seed=seed,
        preprocessing=config["preprocessing"],
    )


def _compare_trees(reference: Any, candidate: Any, *, label: str, atol: float, rtol: float):
    import jax

    reference_leaves = jax.tree_util.tree_leaves(reference)
    candidate_leaves = jax.tree_util.tree_leaves(candidate)
    if len(reference_leaves) != len(candidate_leaves):
        raise AssertionError(f"patched Octo {label} tree differs from the official model")
    worst = {"max_abs_diff": 0.0, "max_reference_abs": 0.0}
    for index, (left, right) in enumerate(zip(reference_leaves, candidate_leaves)):
        left, right = np.asarray(left), np.asarray(right)
        if left.dtype.kind not in "biufc":
            continue
        if left.shape != right.shape:
            raise AssertionError(f"zero-adapter {label} leaf {index} changed shape")
        if left.size == 0:
            continue
        if left.dtype.kind in "biu":
            if not np.array_equal(left, right):
                raise AssertionError(f"zero-adapter equivalence failed at {label} mask leaf {index}")
            continue
        difference = float(np.max(np.abs(left - right)))
        worst["max_abs_diff"] = max(worst["max_abs_diff"], difference)
        worst["max_reference_abs"] = max(worst["max_reference_abs"], float(np.max(np.abs(left))))
        if not np.allclose(left, right, atol=atol, rtol=rtol):
            raise AssertionError(
                f"zero-adapter equivalence failed at {label} leaf {index}; "
                f"max abs diff={difference}, max |reference|={float(np.max(np.abs(left)))}"
            )
    return worst


def assert_zero_adapter_equivalence(
    bundle: ResearchOctoBundle, *, atol: float = 1e-5, rtol: float = 1e-5
) -> dict[str, dict[str, float]]:
    """Verify that zero adapters change neither transformer outputs nor sampled actions.

    Both graphs run in true float32. They still fuse into different kernels, so
    float32 summation order differs; on GPU that leaves a few-ulp residue
    (~3e-6 measured on Blackwell). A functional error is orders of magnitude
    larger, so the tolerance is relative to output magnitude. Sampled actions
    are compared with the identical random key that produced the reference
    before the patch was installed.
    """

    _, jax, _, _, _, _, _, _ = _research_imports()
    from .training import initial_adapter_state, materialize_policy_params

    observations = bundle.pretrained_model.example_batch["observation"]
    tasks = bundle.pretrained_model.example_batch["task"]
    timestep_mask = observations["timestep_pad_mask"]
    activate_research_bundle(bundle)
    with true_float32_matmul():
        params = materialize_policy_params(bundle, initial_adapter_state(bundle))
        patched_model = bundle.research_model.replace(params=params)
        patched_outputs = patched_model.run_transformer(
            observations, tasks, timestep_mask, train=False
        )
        patched_actions = patched_model.sample_actions(
            observations,
            tasks,
            unnormalization_statistics=unnormalization_statistics(bundle.action_convention),
            rng=jax.random.PRNGKey(bundle.equivalence_seed),
        )
    report = {
        "transformer_outputs": _compare_trees(
            bundle.reference_transformer_outputs,
            jax.device_get(patched_outputs),
            label="transformer output",
            atol=atol,
            rtol=rtol,
        ),
        "sampled_actions": _compare_trees(
            bundle.reference_sampled_actions,
            jax.device_get(patched_actions),
            label="sampled action",
            atol=atol,
            rtol=rtol,
        ),
    }
    report["sampled_actions"]["random_key_seed"] = float(bundle.equivalence_seed)
    return report


def assert_base_parameters_unchanged(bundle: ResearchOctoBundle) -> dict[str, str]:
    """Fail if any pretrained leaf differs from the pinned checkpoint hash.

    Checked after zoo training and again after an adapter is reloaded: training
    is purely functional over adapter state, so both the held pretrained tree and
    the base leaves inside the research tree must still hash to ``base_sha256``.
    """

    observed = {
        "pretrained_tree": sha256_array_tree(bundle.pretrained_model.params),
        "research_tree_base_leaves": base_parameter_sha256(bundle.research_model.params),
    }
    for name, digest in observed.items():
        if digest != bundle.base_sha256:
            raise AssertionError(
                f"frozen Octo parameters changed ({name}): {digest} != {bundle.base_sha256}"
            )
    return {"base_sha256": bundle.base_sha256, **observed}


def adapter_parameter_paths(params: Any) -> tuple[tuple[str, ...], ...]:
    """Return all patched adapter leaf paths in deterministic order."""

    try:
        import flax.traverse_util
        from flax.core import unfreeze
    except ImportError as exc:
        raise RuntimeError("Flax is required to inspect Octo adapter parameters") from exc
    flat = flax.traverse_util.flatten_dict(unfreeze(params))
    return tuple(sorted(path for path in flat if is_adapter_path(path)))


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
    from ..contracts import PLACEMENT_PRE_BLOCK, AdapterEntry, AdapterSpec, Component

    flat = flax.traverse_util.flatten_dict(unfreeze(bundle.research_model.params))
    entries_and_factors = []
    for path, value in flat.items():
        if path[-1] != "kernel":
            continue
        matched_index = None
        match = None
        for index, part in enumerate(path):
            candidate = ADAPTER_LEAF.fullmatch(str(part))
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
        logical_path = path_string(
            path[:matched_index] + (str(path[matched_index]).removesuffix("_down"),)
        )
        entry = AdapterEntry(
            component=Component(match.group(1)),
            layer=int(match.group(2)),
            parameter_path=logical_path,
            input_dim=int(down.shape[0]),
            output_dim=int(up.shape[1]),
            rank=int(down.shape[1]),
            site=SITE_BLOCK_INPUT,
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
            site=SITE_HEAD_KERNEL,
        )
        entries_and_factors.append((entry, (down, up)))

    if not entries_and_factors:
        raise ValueError("no research adapters found in the Octo parameter tree")
    entries_and_factors.sort(
        key=lambda item: (
            item[0].component.value,
            item[0].site,
            item[0].layer,
            item[0].parameter_path,
        )
    )
    spec = AdapterSpec(
        base_model_id=OCTO_MODEL_ID,
        base_revision=bundle.base_revision,
        base_sha256=bundle.base_sha256,
        alpha=bundle.adapter_alpha,
        token_width=token_width,
        entries=tuple(item[0] for item in entries_and_factors),
        architecture_sha256=bundle.architecture_sha256,
        token_routing=tuple(sorted(bundle.token_routing.items())),
        action_convention=bundle.action_convention,
        placement=PLACEMENT_PRE_BLOCK,
    )
    return spec, {entry.parameter_path: pair for entry, pair in entries_and_factors}
