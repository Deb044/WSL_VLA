from __future__ import annotations

import numpy as np

from wsl_vla.evaluation.ood import OODReferenceBank, ReferenceAdapter
from wsl_vla.evaluation.ood_bank import (
    OODReferenceBankArtifact,
    load_reference_bank,
    save_reference_bank,
)


def test_ood_reference_bank_round_trip_is_no_pickle_and_fold_bound(tmp_path):
    samples = []
    suites = ("libero_spatial", "libero_object", "libero_goal")
    for index in range(30):
        suite = suites[index // 10]
        samples.append(
            ReferenceAdapter(
                task_id=f"{suite}_{index % 10}",
                suite=suite,
                evidence={
                    name: np.asarray([index], dtype=np.float32)
                    for name in ("vision", "language", "action")
                },
                latents={
                    name: np.full((2, 3), index, dtype=np.float32)
                    for name in ("vision", "language", "action")
                },
            )
        )
    artifact = OODReferenceBankArtifact(
        bank=OODReferenceBank(samples, held_out_suite="libero_10"),
        base_sha256="a" * 64,
        adapter_spec_sha256="b" * 64,
        alignment_checkpoint_sha256="c" * 64,
        source_sample_count=270,
    )
    path = tmp_path / "bank.npz"
    save_reference_bank(artifact, path)
    loaded = load_reference_bank(path)
    assert loaded.base_sha256 == "a" * 64
    assert loaded.alignment_checkpoint_sha256 == "c" * 64
    assert len(loaded.bank.samples) == 30
    assert loaded.bank.training_suites == tuple(sorted(suites))
    expected = {sample.task_id: sample for sample in artifact.bank.samples}
    observed = {sample.task_id: sample for sample in loaded.bank.samples}
    assert np.array_equal(
        observed["libero_spatial_4"].latents["vision"],
        expected["libero_spatial_4"].latents["vision"],
    )
