from __future__ import annotations

from wsl_vla.contracts import ComponentSwapRecord, OODEvaluationRecord
from wsl_vla.evaluation.records import (
    append_component_swap_record,
    append_ood_evaluation_record,
    load_component_swap_records,
    load_ood_evaluation_records,
)


def test_ood_and_component_swap_jsonl_round_trip(tmp_path):
    ood = OODEvaluationRecord(
        run_id="run",
        held_out_suite="libero_10",
        task_id="libero_10_0",
        seed=17,
        method="mapped_zero_shot",
        adaptation_steps=0,
        rollout_count=2,
        successes=1,
        checkpoint_sha256="a" * 64,
        wall_time_seconds=1.0,
        initialization_indices=(3, 4),
        rollout_seeds=(30, 31),
        source_training_suites=("libero_spatial", "libero_object", "libero_goal"),
        alignment_checkpoint_sha256="b" * 64,
    )
    ood_path = tmp_path / "ood.jsonl"
    append_ood_evaluation_record(ood, ood_path)
    restored_ood = load_ood_evaluation_records(ood_path)[0]
    assert restored_ood.success_rate == 0.5
    assert restored_ood.initialization_indices == [3, 4]

    swap = ComponentSwapRecord(
        run_id="run",
        suite="libero_10",
        seed=17,
        condition="proposed_asymmetric",
        transition_stage=1,
        evaluated_task_id="libero_10_0",
        swap_condition="vision",
        rollout_count=2,
        baseline_successes=2,
        swapped_successes=1,
        initialization_indices=(3, 4),
        previous_checkpoint_sha256="a" * 64,
        current_checkpoint_sha256="b" * 64,
        swapped_checkpoint_sha256="c" * 64,
        latent_drift={"vision": 1.0, "language": 0.5, "action": 0.2},
        rollout_seeds=(30, 31),
        baseline_wall_time_seconds=1.0,
        swapped_wall_time_seconds=1.0,
    )
    swap_path = tmp_path / "swaps.jsonl"
    append_component_swap_record(swap, swap_path)
    restored_swap = load_component_swap_records(swap_path)[0]
    assert restored_swap.success_rate_drop == 0.5
