#!/usr/bin/env python3
"""Checkpoint verification, zero-adapter equivalence, and gradient gates.

Run profile 1. Loads the pinned Octo-Small checkpoint, installs the
token-conditioned pre-block adapters, and requires that:

* zero adapters leave transformer outputs *and* sampled actions (identical
  random key) unchanged;
* every adapter branch is trainable - non-zero ``up`` gradients immediately,
  and non-zero ``down`` gradients once the ``up`` factors have moved;
* the frozen base still hashes to the pinned checkpoint afterwards.

It also records the independent checkpoint, architecture, preprocessing, and
controller hashes, the token routing, and which observations the checkpoint
actually consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.experiments.protocol import load_yaml, validate_research_config
from wsl_vla.experiments.runtime import environment_record
from wsl_vla.octo.bridge import (
    adapter_parameter_paths,
    assert_base_parameters_unchanged,
    assert_zero_adapter_equivalence,
    build_adapter_spec_and_factors,
    load_research_octo_from_config,
    trace_adapter_routing,
)
from wsl_vla.octo.training import assert_adapter_gradient_reachability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="research_results/preflight/octo_gates.json"
    )
    args = parser.parse_args()

    import jax

    config = load_yaml(args.config)
    validate_research_config(config)
    environment = environment_record()
    bundle = load_research_octo_from_config(config, args.checkpoint, seed=args.seed)
    equivalence = assert_zero_adapter_equivalence(bundle)
    example = bundle.pretrained_model.example_batch
    batch = {
        "observation": example["observation"],
        "task": example["task"],
        "action": example["action"],
        "action_pad_mask": np.ones_like(np.asarray(example["action"]), dtype=bool),
    }
    gradients = assert_adapter_gradient_reachability(bundle, batch, jax.random.PRNGKey(args.seed))
    base = assert_base_parameters_unchanged(bundle)
    # Read the routing back from the live graph, not from the table it was built from.
    routing = trace_adapter_routing(bundle)
    routed = set(bundle.token_routing.values())
    for route, count in routing["token_counts"].items():
        if (count > 0) != (route in routed):
            raise AssertionError(
                f"the live graph routes {count} tokens to {route!r}, which disagrees with the "
                f"routing table {dict(bundle.token_routing)}"
            )
    if any(float(abs(value).max()) != 0.0 for value in routing["residuals"].values()):
        raise AssertionError("zero-initialized adapters add a non-zero residual")
    paths = adapter_parameter_paths(bundle.research_model.params)
    spec, _ = build_adapter_spec_and_factors(
        bundle, token_width=int(config["adapter"]["token_width"])
    )
    report = {
        "ok": True,
        "environment": environment,
        "identity": bundle.identity(),
        "adapter_spec_sha256": spec.sha256(),
        "adapter_placement": spec.placement,
        "adapter_entries": len(spec.entries),
        "transformer_adapter_leaves": len(paths),
        "diffusion_kernel_count": len(bundle.diffusion_kernel_paths),
        "zero_adapter_equivalence": {**equivalence, "atol": 1e-5, "rtol": 1e-5,
                                     "matmul_precision": "float32"},
        "gradient_reachability": gradients,
        "token_routing_in_live_graph": {
            "token_counts": routing["token_counts"],
            "adapted_blocks": len(routing["residuals"]),
        },
        "base_parameters_unchanged": base,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "ok": True,
        "output": str(output),
        "profile": environment["profile"],
        "base_sha256": bundle.base_sha256,
        "architecture_sha256": bundle.architecture_sha256,
        "zero_adapter_equivalence": equivalence,
        "adapter_branches": gradients["branches"],
        "tokens_per_route": routing["token_counts"],
        "minimum_initial_up_gradient_norm": gradients["minimum_initial_up_gradient_norm"],
        "maximum_initial_down_gradient_norm": gradients["maximum_initial_down_gradient_norm"],
        "minimum_post_update_down_gradient_norm": gradients["minimum_post_update_down_gradient_norm"],
        "proprio_consumed": bundle.preprocessing.proprio_consumed,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
