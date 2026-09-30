from __future__ import annotations

import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.research


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX research extra not installed")
def test_task_loss_produces_nonzero_gradients_for_all_modalities():
    import jax
    import jax.numpy as jnp

    from wsl_vla.alignment.models import assert_task_loss_gradients, refine_latents

    initial = {
        "vision": jnp.ones((2, 3)),
        "language": jnp.ones((2, 3)) * 2,
        "action": jnp.ones((2, 3)) * 3,
    }

    def task_loss(latents, batch):
        decoded = latents["vision"] + 2 * latents["language"] + 3 * latents["action"]
        return jnp.mean(jnp.square(decoded - batch))

    gradients = assert_task_loss_gradients(
        initial, task_loss=lambda latents: task_loss(latents, jnp.zeros((2, 3)))
    )
    assert all(np.linalg.norm(np.asarray(value)) > 0 for value in gradients.values())

    refined, history = refine_latents(
        initial,
        task_loss=task_loss,
        batches=[jnp.zeros((2, 3))],
        gammas={"vision": 0.0, "language": 0.0, "action": 0.0},
        steps=2,
        learning_rate=0.01,
    )
    assert len(history) == 2
    assert all(not np.array_equal(np.asarray(refined[name]), np.asarray(initial[name])) for name in initial)

    with pytest.raises(AssertionError, match="detached from action"):
        assert_task_loss_gradients(
            initial,
            task_loss=lambda latents: jnp.sum(latents["vision"] + latents["language"]),
        )


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX research extra not installed")
def test_latent_refinement_early_stops_and_restores_best_validation_state():
    import jax.numpy as jnp

    from wsl_vla.alignment.models import refine_latents

    initial = {name: jnp.asarray([1.0]) for name in ("vision", "language", "action")}

    def train_loss(latents, batch):
        return sum(jnp.square(value - batch).sum() for value in latents.values())

    refined, history = refine_latents(
        initial,
        task_loss=train_loss,
        batches=[jnp.asarray([0.0])],
        gammas={name: 0.0 for name in initial},
        steps=20,
        learning_rate=0.1,
        validation_loss=lambda latents, batch: jnp.asarray(1.0),
        validation_batches=[jnp.asarray([0.0])],
        early_stopping_patience=2,
    )
    assert len(history) == 3
    assert all(float(refined[name][0]) == pytest.approx(0.8) for name in initial)
