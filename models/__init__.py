"""
models/__init__.py

WSL-VLA Models: Official Octo-Small 1.5 Integration, Modality Adapters,
Weight Autoencoders, Multi-Positive Contrastive Alignment, Differential Regularizers,
and Linear Ridge Prompt Mappers.
"""
from models.alignment_eval import (
    AlignmentGateResult,
    alignment_advantage_gate,
    multi_positive_retrieval_accuracy,
)
from models.component_analysis import (
    component_drift,
    swap_effective_updates,
    validate_consecutive_stages,
)
from models.contrastive_alignment import (
    AlignmentSystem,
    DeepSetsEvidenceEncoder,
    EvidenceProjector,
    multi_positive_info_nce,
)
from models.differential_regularizer import (
    EmpiricalShell,
    assert_task_loss_gradients,
    differential_local_penalty,
    estimate_empirical_shell,
    project_to_empirical_shell,
    refine_latents,
)
from models.evidence_extractor import (
    masked_action_statistics,
    pad_feature_set,
)
from models.flax_adapters import (
    apply_dense_kernel_updates,
    apply_external_adapters,
    discover_diffusion_kernel_paths,
    initialize_external_adapters,
    path_string,
)
from models.latent_adapter import (
    decoded_token_task_loss,
    effective_updates_to_state,
    unpack_effective_tokens_jax,
)
from models.octo_model import (
    OCTO_GIT_REVISION,
    OCTO_MODEL_ID,
    OCTO_MODEL_REVISION,
    ResearchOctoBundle,
    adapter_parameter_paths,
    assert_zero_adapter_equivalence,
    build_adapter_spec_and_factors,
    install_octo_modality_patch,
    load_official_octo,
    load_official_octo_research_model,
    load_research_octo,
)
from models.octo_training import (
    adapter_value_and_grad,
    apply_internal_adapter_params,
    extract_decoded_transformer_params,
    extract_internal_adapter_params,
    initial_adapter_state,
    materialize_policy_params,
    octo_diffusion_loss,
)
from models.packing import (
    PackedAdapter,
    effective_update,
    load_packed_adapter,
    pack_low_rank_adapter,
    save_packed_adapter,
    unpack_effective_updates,
)
from models.prompt_mapper import (
    LinearRidgeMapper,
    fit_linear_ridge_mapper,
)
from models.weight_autoencoder import (
    PackedWeightAutoencoder,
    PackedWeightDecoder,
    PackedWeightEncoder,
    masked_reconstruction_loss,
    require_jax,
)

__all__ = [
    "AlignmentGateResult",
    "AlignmentSystem",
    "DeepSetsEvidenceEncoder",
    "EmpiricalShell",
    "EvidenceProjector",
    "LinearRidgeMapper",
    "OCTO_GIT_REVISION",
    "OCTO_MODEL_ID",
    "OCTO_MODEL_REVISION",
    "PackedAdapter",
    "PackedWeightAutoencoder",
    "PackedWeightDecoder",
    "PackedWeightEncoder",
    "ResearchOctoBundle",
    "adapter_parameter_paths",
    "adapter_value_and_grad",
    "alignment_advantage_gate",
    "apply_dense_kernel_updates",
    "apply_external_adapters",
    "apply_internal_adapter_params",
    "assert_task_loss_gradients",
    "assert_zero_adapter_equivalence",
    "build_adapter_spec_and_factors",
    "component_drift",
    "decoded_token_task_loss",
    "differential_local_penalty",
    "discover_diffusion_kernel_paths",
    "effective_update",
    "effective_updates_to_state",
    "estimate_empirical_shell",
    "extract_decoded_transformer_params",
    "extract_internal_adapter_params",
    "fit_linear_ridge_mapper",
    "initial_adapter_state",
    "initialize_external_adapters",
    "install_octo_modality_patch",
    "load_official_octo",
    "load_official_octo_research_model",
    "load_packed_adapter",
    "load_research_octo",
    "masked_action_statistics",
    "masked_reconstruction_loss",
    "materialize_policy_params",
    "multi_positive_info_nce",
    "multi_positive_retrieval_accuracy",
    "octo_diffusion_loss",
    "pack_low_rank_adapter",
    "pad_feature_set",
    "path_string",
    "project_to_empirical_shell",
    "refine_latents",
    "require_jax",
    "save_packed_adapter",
    "swap_effective_updates",
    "unpack_effective_tokens_jax",
    "unpack_effective_updates",
    "validate_consecutive_stages",
]
