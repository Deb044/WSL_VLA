from __future__ import annotations

import copy

import numpy as np
import pytest

from tests.synthetic import synthetic_spec
from wsl_vla.alignment.mapping import fit_linear_ridge_mapper
from wsl_vla.contracts import (
    ADAPTER_SPEC_SCHEMA_VERSION,
    LIBERO_ACTION_CONVENTION,
    PLACEMENT_PRE_BLOCK,
    AdapterSpec,
    LegacyArtifactError,
    require_schema,
)
from wsl_vla.experiments.manifest import load_experiment_manifest
from wsl_vla.experiments.protocol import (
    RESEARCH_CONFIG_SCHEMA_VERSION,
    load_yaml,
    publication_folds,
    validate_reference_tasks,
    validate_research_config,
)


def test_adapter_spec_roundtrip_keeps_every_identity_field():
    spec = synthetic_spec()
    restored = AdapterSpec.from_dict(spec.to_dict())

    assert restored == spec and restored.sha256() == spec.sha256()
    assert restored.schema_version == ADAPTER_SPEC_SCHEMA_VERSION
    assert restored.placement == PLACEMENT_PRE_BLOCK
    assert restored.action_convention == LIBERO_ACTION_CONVENTION


def test_version_one_adapter_spec_is_inspectable_but_rejected():
    legacy = synthetic_spec().to_dict()
    legacy["schema_version"] = 1
    with pytest.raises(LegacyArtifactError):
        AdapterSpec.from_dict(legacy)
    with pytest.raises(LegacyArtifactError):
        require_schema(1, ADAPTER_SPEC_SCHEMA_VERSION, "adapter spec")
    with pytest.raises(ValueError):
        require_schema(ADAPTER_SPEC_SCHEMA_VERSION + 1, ADAPTER_SPEC_SCHEMA_VERSION, "adapter spec")


def test_linear_mapper_recovers_affine_relation():
    rng = np.random.default_rng(9)
    evidence = rng.normal(size=(30, 4))
    weight = rng.normal(size=(4, 12))
    bias = rng.normal(size=(12,))
    latents = (evidence @ weight + bias).reshape(30, 3, 4)
    mapper = fit_linear_ridge_mapper(evidence, latents, ridge_alpha=1e-10)
    np.testing.assert_allclose(mapper.predict(evidence), latents, atol=1e-7)


def test_locked_protocol_population_comes_from_the_full_manifest():
    config = load_yaml("configs/research/base.yaml")
    tasks = load_yaml("configs/reference_tasks.yaml")
    validate_research_config(config)
    validate_reference_tasks(tasks)
    manifest = load_experiment_manifest(
        "configs/research/manifests/full.yaml", reference_tasks=tasks
    )

    counts = manifest.derived_counts()
    assert counts["independent_runs"] == 120
    assert counts["adapter_samples"] == 360
    assert len(manifest.folds) == len(publication_folds()) == 4
    assert all(len(fold.train_suites) == 3 for fold in publication_folds())
    assert {fold.held_out_suite for fold in manifest.folds} == {
        fold.test_suite for fold in publication_folds()
    }


def test_research_config_is_versioned_and_complete():
    config = load_yaml("configs/research/base.yaml")
    assert config["schema_version"] == RESEARCH_CONFIG_SCHEMA_VERSION

    legacy = copy.deepcopy(config)
    legacy["schema_version"] = 1
    with pytest.raises(ValueError):
        validate_research_config(legacy)

    for section, key in (("alignment", "window_chunk"), ("preprocessing", "image_transform")):
        broken = copy.deepcopy(config)
        del broken[section][key]
        with pytest.raises(ValueError, match=key):
            validate_research_config(broken)

    # Counts live in the experiment manifest, never in the shared configuration.
    assert "population" not in config
