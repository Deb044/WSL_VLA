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


def test_directory_digest_is_order_independent_and_path_sensitive(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a").write_bytes(b"one")
    (left / "b").write_bytes(b"two")
    (right / "b").write_bytes(b"two")
    (right / "a").write_bytes(b"one")
    assert provenance.sha256_directory(left) == provenance.sha256_directory(right)
    (right / "a").rename(right / "c")
    assert provenance.sha256_directory(left) != provenance.sha256_directory(right)


def _git_checkout(root):
    import subprocess

    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def test_editable_install_reads_clean_checkout_commit(monkeypatch, tmp_path):
    commit = _git_checkout(tmp_path)
    (tmp_path / "libero.egg-info").mkdir()  # untracked build output is allowed
    payload = json.dumps({"dir_info": {"editable": True}, "url": tmp_path.as_uri()})
    monkeypatch.setattr(
        provenance.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(payload),
    )
    assert provenance.assert_installed_vcs_revision("libero", commit) == commit


def test_editable_install_rejects_modified_checkout(monkeypatch, tmp_path):
    _git_checkout(tmp_path)
    (tmp_path / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    payload = json.dumps({"dir_info": {"editable": True}, "url": tmp_path.as_uri()})
    monkeypatch.setattr(
        provenance.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(payload),
    )
    with pytest.raises(RuntimeError, match="invalid direct_url|modified tracked files"):
        provenance.installed_vcs_commit("libero")


class LegacyEditableDistribution:
    """A ``setup.py develop`` install: metadata in the checkout, no direct_url.json."""

    def __init__(self, checkout):
        self._path = checkout / "libero.egg-info"
        self._path.mkdir()

    def read_text(self, name):
        assert name == "direct_url.json"
        return None


def test_legacy_editable_install_is_identified_by_its_clean_checkout(monkeypatch, tmp_path):
    commit = _git_checkout(tmp_path)
    distribution = LegacyEditableDistribution(tmp_path)
    monkeypatch.setattr(
        provenance.importlib.metadata, "distribution", lambda name: distribution
    )

    assert provenance.assert_installed_vcs_revision("libero", commit) == commit
    with pytest.raises(RuntimeError, match="revision mismatch"):
        provenance.assert_installed_vcs_revision("libero", "b" * 40)
    # Local edits to tracked files make the recorded commit meaningless.
    (tmp_path / "tracked.py").write_text("x = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified tracked files"):
        provenance.installed_vcs_commit("libero")


def test_install_without_any_provenance_is_still_rejected(monkeypatch, tmp_path):
    class Installed:
        _path = tmp_path / "site-packages" / "libero-0.1.dist-info"

        def read_text(self, name):
            return None

    monkeypatch.setattr(provenance.importlib.metadata, "distribution", lambda name: Installed())
    with pytest.raises(RuntimeError, match="provenance-bearing VCS URL"):
        provenance.installed_vcs_commit("libero")

    # An .egg-info that does not sit in a git checkout proves nothing either.
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "libero.egg-info").mkdir()

    class Orphan(Installed):
        _path = orphan / "libero.egg-info"

    monkeypatch.setattr(provenance.importlib.metadata, "distribution", lambda name: Orphan())
    with pytest.raises(RuntimeError, match="provenance-bearing VCS URL"):
        provenance.installed_vcs_commit("libero")
