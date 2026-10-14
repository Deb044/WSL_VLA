"""Loss formulas against explicit reference calculations."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from wsl_vla.alignment.models import (  # noqa: E402
    bidirectional_token_contrastive_loss,
    component_balanced_reconstruction_loss,
    map_windows,
)

pytestmark = pytest.mark.research


def reference_contrastive(latents, valid, evidence, labels, temperature):
    """Direct transcription of the documented formula with Python loops."""

    z = latents / np.linalg.norm(latents, axis=-1, keepdims=True)
    e = evidence / np.linalg.norm(evidence, axis=-1, keepdims=True)
    batch, tokens, _ = z.shape
    score = np.einsum("btd,kd->btk", z, e) / temperature
    forward = []
    for b in range(batch):
        for t in range(tokens):
            if not valid[b, t]:
                continue
            numerator = sum(np.exp(score[b, t, k]) for k in range(batch) if labels[k] == labels[b])
            denominator = sum(np.exp(score[b, t, k]) for k in range(batch))
            forward.append(-np.log(numerator / denominator))
    reverse = []
    for k in range(batch):
        for t in range(tokens):
            candidates = [b for b in range(batch) if valid[b, t]]
            positives = [b for b in candidates if labels[b] == labels[k]]
            if not positives:
                continue
            numerator = sum(np.exp(score[b, t, k]) for b in positives)
            denominator = sum(np.exp(score[b, t, k]) for b in candidates)
            reverse.append(-np.log(numerator / denominator))
    return 0.5 * (np.mean(forward) + np.mean(reverse))


def contrastive_inputs(seed=0, batch=6, tokens=5, dim=4):
    rng = np.random.default_rng(seed)
    latents = rng.normal(size=(batch, tokens, dim))
    evidence = rng.normal(size=(batch, dim))
    labels = np.array([0, 0, 1, 1, 2, 2])
    valid = np.ones((batch, tokens), dtype=bool)
    return latents, valid, evidence, labels


def test_contrastive_loss_matches_the_explicit_reference():
    latents, valid, evidence, labels = contrastive_inputs()
    valid[:, 4] = False       # a layout padding position
    valid[1, 3] = False       # a per-sample hole exercises the general mask
    expected = reference_contrastive(latents, valid, evidence, labels, 0.2)
    observed = bidirectional_token_contrastive_loss(
        jnp.asarray(latents), jnp.asarray(valid), jnp.asarray(evidence), jnp.asarray(labels),
        temperature=0.2,
    )
    assert float(observed) == pytest.approx(expected, rel=1e-5)


def test_contrastive_loss_ignores_padding_in_both_directions():
    latents, valid, evidence, labels = contrastive_inputs(seed=1)
    valid[:, 3:] = False
    args = (jnp.asarray(valid), jnp.asarray(evidence), jnp.asarray(labels))
    clean = bidirectional_token_contrastive_loss(jnp.asarray(latents), *args, temperature=0.1)
    garbage = latents.copy()
    garbage[:, 3:] = 1e3 * np.random.default_rng(9).normal(size=garbage[:, 3:].shape)
    dirty = bidirectional_token_contrastive_loss(jnp.asarray(garbage), *args, temperature=0.1)
    assert float(clean) == pytest.approx(float(dirty), rel=1e-6)
    gradient = jax.grad(
        lambda value: bidirectional_token_contrastive_loss(value, *args, temperature=0.1)
    )(jnp.asarray(garbage))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert not np.asarray(gradient)[:, 3:].any()  # padded tokens receive no gradient


def test_contrastive_loss_is_token_wise_in_the_evidence_to_weight_direction():
    # Both samples have the same mean-pooled latent, but their tokens are swapped
    # across positions. The legacy reverse term pooled tokens first, so it saw
    # two identical candidates and returned exactly chance (log 2). The
    # token-wise term contrasts tokens position by position and does not.
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    latents = np.stack([np.stack([a, b]), np.stack([b, a])])
    evidence = np.stack([a, b])
    labels = np.array([0, 1])
    valid = np.ones((2, 2), dtype=bool)
    temperature = 0.5
    observed = float(
        bidirectional_token_contrastive_loss(
            jnp.asarray(latents), jnp.asarray(valid), jnp.asarray(evidence), jnp.asarray(labels),
            temperature=temperature,
        )
    )
    assert observed == pytest.approx(
        reference_contrastive(latents, valid, evidence, labels, temperature), rel=1e-5
    )
    score = np.einsum("btd,kd->btk", latents, evidence) / temperature
    forward = np.mean(-np.log(np.exp(score)[[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]] / np.exp(score).sum(-1).ravel()))
    legacy_pooled = 0.5 * (forward + np.log(2.0))
    # Position 0 is aligned and position 1 is anti-aligned; their mean is not log 2.
    aligned = -np.log(np.exp(2.0) / (np.exp(2.0) + 1.0))
    anti_aligned = -np.log(1.0 / (np.exp(2.0) + 1.0))
    assert observed == pytest.approx(0.5 * (forward + 0.5 * (aligned + anti_aligned)), rel=1e-5)
    assert abs(observed - legacy_pooled) > 0.1


def test_same_task_samples_are_positives_not_false_negatives():
    latents, valid, evidence, _ = contrastive_inputs(seed=2)
    evidence[1] = evidence[0]  # two samples of one task share its evidence
    args = (jnp.asarray(latents), jnp.asarray(valid), jnp.asarray(evidence))
    grouped = float(
        bidirectional_token_contrastive_loss(*args, jnp.asarray([0, 0, 1, 2, 3, 4]), temperature=0.1)
    )
    separate = float(
        bidirectional_token_contrastive_loss(*args, jnp.asarray([0, 5, 1, 2, 3, 4]), temperature=0.1)
    )
    assert grouped < separate


def reference_reconstruction(reconstruction, target, mask, component_ids):
    losses = []
    for component in range(3):
        selected = component_ids == component
        error = np.square(reconstruction[:, selected] - target[:, selected]) * mask[selected]
        losses.append(error.sum() / (mask[selected].sum() * reconstruction.shape[0]))
    return float(np.mean(losses)), losses


def test_reconstruction_is_averaged_within_then_equally_across_components():
    rng = np.random.default_rng(3)
    component_ids = np.array([0] * 2 + [1] * 3 + [2] * 20)  # action has 10x more tokens
    mask = rng.random((25, 4)) > 0.3
    mask[:, 0] = True
    target = rng.normal(size=(2, 25, 4))
    reconstruction = target + rng.normal(size=target.shape)
    expected, per_component = reference_reconstruction(reconstruction, target, mask, component_ids)
    observed, parts = component_balanced_reconstruction_loss(
        jnp.asarray(reconstruction), jnp.asarray(target), jnp.asarray(mask), component_ids
    )
    assert float(observed) == pytest.approx(expected, rel=1e-5)
    for index, name in enumerate(("vision", "language", "action")):
        assert float(parts[name]) == pytest.approx(per_component[index], rel=1e-5)
    # Errors placed only in masked entries contribute nothing.
    hidden = target + 100.0 * (~mask)
    zero, _ = component_balanced_reconstruction_loss(
        jnp.asarray(hidden), jnp.asarray(target), jnp.asarray(mask), component_ids
    )
    assert float(zero) == 0.0
    # An error confined to the small vision stream weighs as much as one third
    # of the objective, however many action tokens exist.
    only_vision = target.copy()
    only_vision[:, component_ids == 0] += 1.0
    value, parts = component_balanced_reconstruction_loss(
        jnp.asarray(only_vision), jnp.asarray(target), jnp.asarray(mask), component_ids
    )
    assert float(value) == pytest.approx(float(parts["vision"]) / 3.0, rel=1e-6)
    assert float(parts["vision"]) == pytest.approx(1.0, rel=1e-6)


def test_chunked_window_mapping_equals_one_pass_including_gradients():
    rng = np.random.default_rng(4)
    values = jnp.asarray(rng.normal(size=(7, 5, 3)))
    scale = jnp.asarray(rng.normal(size=(7, 5, 1)))

    def function(a, b):
        return jnp.tanh(a * b).sum(axis=-1, keepdims=True) * a

    direct = function(values, scale)
    for chunk in (2, 3, 7, 100):
        np.testing.assert_allclose(
            map_windows(function, (values, scale), chunk=chunk), direct, rtol=1e-6, atol=1e-6
        )
    gradient = jax.grad(lambda a: map_windows(function, (a, scale), chunk=3).sum())(values)
    reference = jax.grad(lambda a: function(a, scale).sum())(values)
    np.testing.assert_allclose(gradient, reference, rtol=1e-5, atol=1e-6)
