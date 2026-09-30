"""
models/differential_regularizer.py

Differential Regularization During Sequential Continual Adaptation.
Enforces modality-asymmetric constraints: gamma_vis, gamma_lang > gamma_act
to protect fragile cognitive features while maintaining motor plasticity.
Constrains latents to the empirical hyperspherical shell (Pi_shell).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from models.weight_autoencoder import require_jax

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


@dataclass(frozen=True)
class EmpiricalShell:
    """Frobenius hyperspherical shell defined by empirical center and radius."""
    center: Any
    radius: Any


def estimate_empirical_shell(latents, valid_mask) -> EmpiricalShell:
    """Compute empirical center and mean radius of valid latent points."""
    require_jax()
    weights = valid_mask.astype(latents.dtype)[..., None]
    count = jnp.maximum(jnp.sum(weights, axis=0), 1)
    center = jnp.sum(latents * weights, axis=0) / count
    distances = jnp.linalg.norm(latents - center[None], axis=-1)
    radius = jnp.sum(distances * valid_mask, axis=0) / jnp.maximum(
        jnp.sum(valid_mask, axis=0), 1
    )
    return EmpiricalShell(center=center, radius=jnp.maximum(radius, 1e-6))


def project_to_empirical_shell(latents, shell: EmpiricalShell):
    """Project latents onto the empirical shell: Pi_shell(z) = center + R * (z - center)/||z - center||."""
    require_jax()
    offset = latents - shell.center
    norm = jnp.linalg.norm(offset, axis=-1, keepdims=True)
    fallback = jnp.zeros_like(offset).at[..., 0].set(1)
    direction = jnp.where(norm > 1e-8, offset / jnp.maximum(norm, 1e-8), fallback)
    return shell.center + shell.radius[..., None] * direction


def differential_local_penalty(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    gammas: Mapping[str, float],
):
    """Asymmetric quadratic penalty: sum_m gamma_m ||z_m^(t) - z_m^(t-1)||^2."""
    require_jax()
    expected = {"vision", "language", "action"}
    if set(current) != expected or set(previous) != expected or set(gammas) != expected:
        raise ValueError("current, previous, and gammas must contain all three modalities")
    return sum(
        gammas[name] * jnp.sum(jnp.square(current[name] - previous[name]))
        for name in sorted(expected)
    )


def assert_task_loss_gradients(
    latents: Mapping[str, Any],
    *,
    task_loss: Callable[[Mapping[str, Any]], Any],
    minimum_norm: float = 1e-12,
):
    """Hard gate: task loss gradients must reach every modality latent."""
    require_jax()
    gradients = jax.grad(task_loss)(latents)
    for name in ("vision", "language", "action"):
        if name not in gradients:
            raise AssertionError(f"task loss has no {name} latent gradient")
        value = gradients[name]
        if not bool(jnp.all(jnp.isfinite(value))):
            raise AssertionError(f"task loss has non-finite {name} gradients")
        if float(jnp.linalg.norm(value)) <= minimum_norm:
            raise AssertionError(f"task loss is detached from {name} latents")
    return gradients


def refine_latents(
    initial: Mapping[str, Any],
    *,
    task_loss: Callable[[Mapping[str, Any], Any], Any],
    batches: list[Any],
    gammas: Mapping[str, float],
    steps: int,
    learning_rate: float,
):
    """Differentiate task loss through decoder and Octo adapter application."""
    require_jax()
    if steps <= 0 or not batches:
        raise ValueError("refinement requires positive steps and at least one batch")
    current = {name: jnp.asarray(value) for name, value in initial.items()}
    previous = {name: jax.lax.stop_gradient(value) for name, value in current.items()}

    def objective(values, batch):
        task = task_loss(values, batch)
        regularizer = differential_local_penalty(values, previous, gammas)
        return task + regularizer, {"task_loss": task, "regularizer": regularizer}

    history = []
    for step in range(steps):
        (loss, auxiliary), gradients = jax.value_and_grad(objective, has_aux=True)(
            current, batches[step % len(batches)]
        )
        current = jax.tree_util.tree_map(
            lambda value, gradient: value - learning_rate * gradient,
            current,
            gradients,
        )
        history.append({"loss": loss, **auxiliary})
    return current, history


__all__ = [
    "EmpiricalShell",
    "assert_task_loss_gradients",
    "differential_local_penalty",
    "estimate_empirical_shell",
    "project_to_empirical_shell",
    "refine_latents",
    "require_jax",
]
