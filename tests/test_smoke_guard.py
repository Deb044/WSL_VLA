from __future__ import annotations

import json

import pytest

from wsl_vla.smoke_guard import require_explicit_smoke_test, write_smoke_marker


def test_proxy_requires_explicit_smoke_acknowledgement():
    with pytest.raises(RuntimeError, match="legacy PyTorch proxy"):
        require_explicit_smoke_test(False)


def test_proxy_cannot_write_to_research_results():
    with pytest.raises(ValueError, match="cannot write inside research_results"):
        require_explicit_smoke_test(True, output_directories=("research_results/run",))


def test_proxy_artifact_directory_is_machine_readably_marked(tmp_path):
    marker = write_smoke_marker(tmp_path / "proxy", command="fixture")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["official_octo"] is False
    assert payload["publication_eligible"] is False
    assert payload["artifact_kind"] == "legacy_pytorch_proxy_smoke_test"
