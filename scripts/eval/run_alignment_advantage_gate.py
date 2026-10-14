#!/usr/bin/env python3
"""Deprecated entry point kept so existing automation does not break.

The "alignment advantage gate" required the aligned checkpoint to beat the
reconstruction-only control before anything could proceed. That conflated a
scientific outcome with infrastructure health. Its replacement,
``scripts/eval/evaluate_alignment_pair.py``, reports integrity, inconclusive
evidence, and negative findings separately and never requires alignment to win.
This shim forwards to it unchanged.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    print(
        "run_alignment_advantage_gate.py is deprecated: forwarding to "
        "evaluate_alignment_pair.py (no must-win gate).",
        file=sys.stderr,
    )
    target = Path(__file__).with_name("evaluate_alignment_pair.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
