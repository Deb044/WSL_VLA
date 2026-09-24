#!/usr/bin/env python3
"""
tests/test_methodology1.py

Unit test suite for Methodology 1:
  - Section 1.1: Component-Factorized Tokenization & Autoencoder (g_phi, h_psi)
  - Section 1.2: Modality-Specific Contrastive Alignment & InfoNCE Losses
  - Section 1.3: Differential Regularization & Hyperspherical Shell Projection
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (
    FactorizedWeightAutoencoder,
    ModalityContrastiveAligner,
    DifferentialRegularizer,
    hyperspherical_shell_projection,
)


def test_factorized_weight_autoencoder():
    print("Test 1: Testing Component-Factorized Weight Autoencoder (g_phi, h_psi)...")
    B, L, r, H = 2, 8, 16, 384
    d_latent = 128
    dummy_delta_w = torch.randn(B, L, 3, r, H)

    autoencoder = FactorizedWeightAutoencoder(num_layers=L, rank=r, hidden_dim=H, d_latent=d_latent)

    # Forward pass
    rec_delta_w, latents = autoencoder(dummy_delta_w)

    assert rec_delta_w.shape == (B, L, 3, r, H), f"Reconstruction shape mismatch: {rec_delta_w.shape}"
    assert latents["vis"].shape == (B, L, d_latent)
    assert latents["lang"].shape == (B, L, d_latent)
    assert latents["act"].shape == (B, L, d_latent)
    assert latents["stacked"].shape == (B, L, 3, d_latent)

    # Verify reconstruction gradient flow
    loss = F.mse_loss(rec_delta_w, dummy_delta_w)
    loss.backward()
    assert autoencoder.encoder.input_proj[0].weight.grad is not None
    print("  -> Success: Factorized autoencoder encodes and decodes [B, 8, 3, 16, 384] with valid gradients.")


def test_modality_contrastive_alignment():
    print("Test 2: Testing Modality-Specific Contrastive Alignment (L_align^(m), tau_m, L_total)...")
    B, L, d_latent = 4, 8, 128
    aligner = ModalityContrastiveAligner(d_latent=d_latent, vis_dim=128, lang_dim=384, act_dim=28, num_classes=4)

    dummy_latents = {
        "vis": torch.randn(B, L, d_latent, requires_grad=True),
        "lang": torch.randn(B, L, d_latent, requires_grad=True),
        "act": torch.randn(B, L, d_latent, requires_grad=True),
    }
    dummy_evidence = {
        "e_vis": torch.randn(B, 128),
        "e_lang": torch.randn(B, 384),
        "e_act": torch.randn(B, 28),
    }
    dummy_rec_w = torch.randn(B, L, 3, 16, 384, requires_grad=True)
    dummy_tgt_w = torch.randn(B, L, 3, 16, 384)
    dummy_labels = torch.tensor([0, 1, 2, 3])

    total_loss, metrics = aligner.compute_total_loss(
        rec_delta_w=dummy_rec_w,
        target_delta_w=dummy_tgt_w,
        latents=dummy_latents,
        evidence=dummy_evidence,
        task_labels=dummy_labels,
    )

    assert total_loss.item() > 0.0
    assert "loss_align_vis" in metrics
    assert "loss_align_lang" in metrics
    assert "loss_align_act" in metrics
    assert "tau_vis" in metrics

    total_loss.backward()
    assert aligner.log_tau_vis.grad is not None
    assert aligner.proj_vis.net[0].weight.grad is not None
    print(f"  -> Success: Bidirectional contrastive losses and learnable temperatures verified. Total Loss: {total_loss.item():.4f}")


def test_hyperspherical_shell_projection():
    print("Test 3: Testing Hyperspherical Shell Projection (Pi_shell)...")
    z = torch.randn(4, 8, 128)
    radius = 2.5
    z_proj = hyperspherical_shell_projection(z, radius=radius)

    # Check Frobenius norm per sample
    norms = torch.norm(z_proj, p="fro", dim=(-2, -1))
    for norm_val in norms:
        assert abs(norm_val.item() - radius) < 1e-4, f"Projection radius mismatch: expected {radius}, got {norm_val.item()}"
    print(f"  -> Success: All projected tensors strictly lie on hyperspherical shell of radius {radius}.")


def test_differential_regularizer():
    print("Test 4: Testing Differential Regularization (gamma_vis, gamma_lang > gamma_act)...")
    reg = DifferentialRegularizer(schedule="fixed", gamma_vis=1.5, gamma_lang=1.5, gamma_act=0.3)
    gammas = reg.get_gammas()
    assert gammas["vis"] == 1.5
    assert gammas["lang"] == 1.5
    assert gammas["act"] == 0.3
    assert gammas["vis"] > gammas["act"]

    z_curr = {
        "vis": torch.randn(1, 8, 128, requires_grad=True),
        "lang": torch.randn(1, 8, 128, requires_grad=True),
        "act": torch.randn(1, 8, 128, requires_grad=True),
    }
    z_init = {
        "vis": torch.randn(1, 8, 128),
        "lang": torch.randn(1, 8, 128),
        "act": torch.randn(1, 8, 128),
    }

    loss_reg, meta = reg.compute_regularization_loss(z_curr, z_init)
    loss_reg.backward()

    assert z_curr["vis"].grad is not None
    assert z_curr["act"].grad is not None
    print(f"  -> Success: Differential Regularization loss computed: {loss_reg.item():.4f} with valid backward pass.")


if __name__ == "__main__":
    print("=" * 80)
    print(" Running Unit Test Suite for Methodology 1")
    print("=" * 80)
    test_factorized_weight_autoencoder()
    test_modality_contrastive_alignment()
    test_hyperspherical_shell_projection()
    test_differential_regularizer()
    print("=" * 80)
    print(" ALL METHODOLOGY 1 UNIT TESTS PASSED!")
    print("=" * 80)
