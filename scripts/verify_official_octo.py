#!/usr/bin/env python3
"""Load pinned Octo-Small and verify zero-adapter functional equivalence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.octo.bridge import (
    OCTO_GIT_REVISION,
    adapter_parameter_paths,
    assert_zero_adapter_equivalence,
    load_research_octo,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="research_results/preflight/octo_equivalence.json"
    )
    parser.add_argument(
        "--matmul-precision",
        default="highest",
        choices=("highest", "default"),
        help="Equivalence is a correctness check, so it runs in true float32. "
        "GPUs otherwise use TF32 for float32 matmuls, and the patched graph "
        "fuses differently, giving ~1e-3 rounding differences.",
    )
    args = parser.parse_args()

    import jax

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    bundle = load_research_octo(
        args.checkpoint,
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )
    assert_zero_adapter_equivalence(bundle)
    paths = adapter_parameter_paths(bundle.research_model.params)
    if not paths:
        raise RuntimeError("patched Octo contains no modality adapter parameters")
    report = {
        "ok": True,
        "octo_git_revision": OCTO_GIT_REVISION,
        "matmul_precision": args.matmul_precision,
        "jax_devices": [str(device) for device in jax.devices()],
        "base_sha256": bundle.base_sha256,
        "base_revision": bundle.base_revision,
        "adapter_parameter_count": len(paths),
        "diffusion_kernel_count": len(bundle.diffusion_kernel_paths),
        "adapter_paths": ["/".join(path) for path in paths],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
