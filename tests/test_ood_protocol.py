from __future__ import annotations

import hashlib

import numpy as np
import pytest

from wsl_vla.evaluation.ood import (
    OOD_METHODS,
    OODCandidate,
    OODReferenceBank,
    ReferenceAdapter,
    run_ood_protocol,
)
from wsl_vla.evaluation.sequential import RolloutOutcome


def reference(task_id, suite, value):
    return ReferenceAdapter(
        task_id=task_id,
        suite=suite,
        evidence={name: np.array([value], dtype=float) for name in ("vision", "language", "action")},
        latents={name: np.full((2, 2), value, dtype=float) for name in ("vision", "language", "action")},
    )


def test_reference_bank_excludes_test_suite_and_provides_real_baselines():
    bank = OODReferenceBank(
        [reference("a", "libero_spatial", 1), reference("b", "libero_goal", 3)],
        held_out_suite="libero_10",
    )
    task_id, latents = bank.nearest_neighbor_latents(
        {name: np.array([2.8]) for name in ("vision", "language", "action")}
    )
    assert task_id == "b"
    assert np.all(latents["vision"] == 3)
    assert np.all(bank.mean_latents()["action"] == 2)
    with pytest.raises(ValueError, match="held-out"):
        OODReferenceBank(
            [reference("leak", "libero_10", 0)], held_out_suite="libero_10"
        )


def test_ood_protocol_enforces_matched_steps_and_identical_initializations():
    builders = {}
    for method in OOD_METHODS:
        steps = 20 if method in {"mapped_latent_refinement", "mapped_weight_finetune"} else 0
        builders[method] = lambda task, instruction, seed, method=method, steps=steps: OODCandidate(
            state=method,
            checkpoint_sha256=hashlib.sha256(method.encode()).hexdigest(),
            adaptation_steps=steps,
        )

    def rollout(state, task_id, count, seed):
        return RolloutOutcome(1, tuple(range(count)), tuple(seed + i for i in range(count)))

    records = run_ood_protocol(
        task_ids=("libero_10_0",),
        instructions=("task",),
        candidate_builders=builders,
        rollout=rollout,
        run_id="run",
        held_out_suite="libero_10",
        seed=17,
        rollout_count=3,
        matched_adaptation_steps=20,
        source_training_suites=("libero_spatial", "libero_object", "libero_goal"),
        alignment_checkpoint_sha256="a" * 64,
    )
    assert tuple(record.method for record in records) == OOD_METHODS
    assert all(record.initialization_indices == (0, 1, 2) for record in records)

    broken = dict(builders)
    broken["mapped_weight_finetune"] = lambda task, instruction, seed: OODCandidate(
        state=None, checkpoint_sha256="b" * 64, adaptation_steps=19
    )
    with pytest.raises(ValueError, match="matched steps"):
        run_ood_protocol(
            task_ids=("libero_10_0",),
            instructions=("task",),
            candidate_builders=broken,
            rollout=rollout,
            run_id="run",
            held_out_suite="libero_10",
            seed=17,
            rollout_count=3,
            matched_adaptation_steps=20,
            source_training_suites=("libero_spatial", "libero_object", "libero_goal"),
            alignment_checkpoint_sha256="a" * 64,
        )
