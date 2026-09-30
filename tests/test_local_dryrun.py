#!/usr/bin/env python3
"""
tests/test_local_dryrun.py

Comprehensive local CPU pre-flight dry-run unit test suite.
Validates the unified VLA factory (build_vla_model), Octo-Small (27M) architecture,
factorized LoRA parameter freezing, tensor slicing into [L, 3, r, H],
multi-modal evidence extraction, and atomic checkpointing.
"""
import os
import sys
import tempfile
import pytest

pytest.importorskip("peft", reason="legacy PyTorch smoke fixture requires the legacy extra")
import torch

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wsl_vla.smoke.legacy_torch.data import LiberoTaskDataset
from wsl_vla.smoke.legacy_torch.models import build_vla_model, TaskEvidenceExtractor


def test_vla_factory_and_octo_small():
    print("Test 1: Testing Unified VLA Factory & Instantiating Octo-Small (27M)...")
    config_octo = {
        "model": {
            "name": "octo_small",
            "num_layers": 8,
            "hidden_dim": 384,
            "action_dim": 7,
            "image_features_dim": 64,
            "language_embed_dim": 384,
        }
    }
    octo_model = build_vla_model(config_octo)
    assert octo_model.num_layers == 8
    assert octo_model.hidden_dim == 384

    # Forward pass test
    vis = torch.randn(2, 64)
    lang = torch.randn(2, 384)
    act_hist = torch.randn(2, 7)
    out_act = octo_model(vis, lang, act_hist)
    assert out_act.shape == (2, 7), f"Bad forward shape: {out_act.shape}"
    print(f"  -> Success: Octo-Small instantiated via factory and forward pass verified.")


def test_octo_small_lora_and_tensor_slicing():
    print("Test 2: Verifying Octo-Small LoRA Parameter Freezing & [8, 3, 16, 384] Slicing...")
    config_octo = {
        "model": {
            "name": "octo_small",
            "num_layers": 8,
            "hidden_dim": 384,
            "action_dim": 7,
            "image_features_dim": 64,
            "language_embed_dim": 384,
        }
    }
    octo_model = build_vla_model(config_octo)
    peft_model = octo_model.attach_factorized_lora(rank=16, alpha=32)

    # Check freezing
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            assert param.requires_grad, f"LoRA param '{name}' must have requires_grad=True"
        else:
            assert not param.requires_grad, f"Backbone param '{name}' must have requires_grad=False"

    # Extract Delta W
    delta_w = octo_model.extract_delta_w()
    expected_shape = (8, 3, 16, 384)
    assert delta_w.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {delta_w.shape}"
    # Verify roundtrip injection
    octo_model.inject_delta_w(delta_w)
    delta_w_injected = octo_model.extract_delta_w()
    assert torch.allclose(delta_w, delta_w_injected, atol=1e-5), "Delta W injection roundtrip failed!"
    print(f"  -> Success: Backbone frozen, LoRA active, Delta W extraction & injection roundtrip verified: {list(delta_w.shape)}.")


def test_small_vla_swapping():
    print("Test 3: Verifying Seamless Model Swapping to SmallVLA...")
    config_small = {
        "model": {
            "name": "small_vla",
            "num_layers": 4,
            "hidden_dim": 256,
            "action_dim": 7,
            "image_features_dim": 64,
            "language_embed_dim": 384,
        }
    }
    small_model = build_vla_model(config_small)
    small_model.attach_factorized_lora(rank=16, alpha=32)
    delta_w = small_model.extract_delta_w()
    assert delta_w.shape == (4, 3, 16, 256)
    print(f"  -> Success: Successfully swapped to SmallVLA [4, 3, 16, 256] via same factory interface.")


def test_evidence_extractor():
    print("Test 4: Extracting Multi-Modal Task Evidence (e_vis, e_lang, e_act)...")
    extractor = TaskEvidenceExtractor(device="cpu", vis_in_dim=64, vis_embed_dim=128)

    dummy_frames = torch.randn(20, 64)
    dummy_text = "pick up the black bowl and place it on the plate"
    dummy_actions = torch.randn(80, 7)

    evidence = extractor.extract_evidence(dummy_frames, dummy_text, dummy_actions)

    assert evidence["e_vis"].shape == (128,), f"Bad e_vis shape: {evidence['e_vis'].shape}"
    assert evidence["e_lang"].shape == (384,), f"Bad e_lang shape: {evidence['e_lang'].shape}"
    assert evidence["e_act"].shape == (28,), f"Bad e_act shape: {evidence['e_act'].shape}"
    print(f"  -> Success: Multi-modal evidence shapes verified.")


def test_atomic_checkpoint_roundtrip():
    print("Test 5: Testing Atomic Checkpoint Serialization & Deserialization...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        from scripts.train_zoo import atomic_save, is_checkpoint_valid

        ckpt_file = os.path.join(tmp_dir, "test_octo_task.pt")
        mock_payload = {
            "task_id": "libero_spatial_0",
            "task_name": "pick_up_bowl",
            "delta_w": torch.randn(8, 3, 16, 384),
            "e_vis": torch.randn(128),
            "e_lang": torch.randn(384),
            "e_act": torch.randn(28),
        }

        # Atomic save
        atomic_save(mock_payload, ckpt_file)
        assert os.path.exists(ckpt_file), "Checkpoint was not created!"
        assert not os.path.exists(ckpt_file + ".tmp"), "Temporary file was not cleaned up!"
        assert is_checkpoint_valid(ckpt_file), "Checkpoint validation failed!"

        loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        assert loaded["delta_w"].shape == (8, 3, 16, 384)
        print("  -> Success: Atomic write, rename, and load verified cleanly for Octo-Small.")


if __name__ == "__main__":
    print("======================================================================")
    print(" Starting Local Pre-Flight Dry-Run Suite (CPU)")
    print("======================================================================")
    test_vla_factory_and_octo_small()
    test_octo_small_lora_and_tensor_slicing()
    test_small_vla_swapping()
    test_evidence_extractor()
    test_atomic_checkpoint_roundtrip()
    print("======================================================================")
    print(" ALL 5 PRE-FLIGHT TESTS PASSED FOR OCTO-SMALL & FACTORY!")
    print("======================================================================")
