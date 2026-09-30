from __future__ import annotations

import json

import pytest

from wsl_vla.experiments import provenance


class FakeDistribution:
    def __init__(self, payload):
        self.payload = payload

    def read_text(self, name):
        assert name == "direct_url.json"
        return self.payload


def test_vcs_revision_is_read_and_compared_from_pep610_metadata(monkeypatch):
    commit = "a" * 40
    payload = json.dumps({"vcs_info": {"vcs": "git", "commit_id": commit}})
    monkeypatch.setattr(
        provenance.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(payload),
    )
    assert provenance.assert_installed_vcs_revision("libero", commit) == commit
    with pytest.raises(RuntimeError, match="revision mismatch"):
        provenance.assert_installed_vcs_revision("libero", "b" * 40)


def test_vcs_revision_rejects_non_vcs_install(monkeypatch):
    monkeypatch.setattr(
        provenance.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(json.dumps({"url": "file:///tmp/libero"})),
    )
    with pytest.raises(RuntimeError, match="invalid direct_url"):
        provenance.installed_vcs_commit("libero")
