"""Hard separation between legacy proxy fixtures and publication artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SMOKE_MARKER = "SMOKE_ONLY.json"


def require_explicit_smoke_test(
    enabled: bool,
    *,
    output_directories: Iterable[str | Path] = (),
) -> None:
    """Reject accidental execution or research-labelled proxy output paths."""

    if not enabled:
        raise RuntimeError(
            "This command uses the legacy PyTorch proxy, not official Octo. "
            "Pass --smoke-test only for plumbing tests; its outputs are not research results."
        )
    for raw_path in output_directories:
        path = Path(raw_path)
        if "research_results" in {part.lower() for part in path.parts}:
            raise ValueError("legacy smoke fixtures cannot write inside research_results")


def write_smoke_marker(output_directory: str | Path, *, command: str) -> Path:
    """Mark a proxy artifact directory so downstream research loaders can reject it."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / SMOKE_MARKER
    marker.write_text(
        json.dumps(
            {
                "artifact_kind": "legacy_pytorch_proxy_smoke_test",
                "official_octo": False,
                "publication_eligible": False,
                "command": command,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker
