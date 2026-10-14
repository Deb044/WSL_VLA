"""Pytest configuration and root path resolution for WSL_VLA."""
from pathlib import Path
import sys


def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file():
            return p
    return Path(__file__).resolve().parent.parent


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def tiny_t5(tmp_path_factory):
    """Offline one-layer T5 configuration for the toy Octo model."""

    pytest.importorskip("octo")
    from tests.octo_tiny import write_tiny_t5

    return write_tiny_t5(tmp_path_factory.mktemp("octo"))


@pytest.fixture(scope="session")
def tiny_bundle(tiny_t5):
    """Toy official Octo wrapped with research adapters (see ``tests/octo_tiny.py``)."""

    from tests.octo_tiny import build_tiny_bundle

    return build_tiny_bundle(tiny_t5)


def _skip_reason(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """State plainly what this session did *not* validate.

    A skipped or deselected test is not a passed test. Tests that need the
    research environment, a GPU, the pinned checkpoint, real demonstrations, or
    the simulator are listed here with the missing prerequisite.
    """

    skipped = terminalreporter.stats.get("skipped", [])
    deselected = [
        item
        for item in terminalreporter.stats.get("deselected", [])
        if hasattr(item, "get_closest_marker") and item.get_closest_marker("integration")
    ]
    integration = [report for report in skipped if "integration" in getattr(report, "keywords", {})]
    other = [report for report in skipped if report not in integration]
    if not (integration or deselected or other):
        return
    terminalreporter.section("not validated in this session (a skip is not a pass)")
    if deselected:
        terminalreporter.write_line(
            f"{len(deselected)} real Octo/LIBERO integration test(s) were not selected. Run "
            "`pytest tests/integration/test_real_pipeline.py -m integration` in a research "
            "environment to validate the real pipeline."
        )
    reasons: dict[str, int] = {}
    for report in integration:
        reasons[_skip_reason(report)] = reasons.get(_skip_reason(report), 0) + 1
    for reason, count in reasons.items():
        terminalreporter.write_line(f"{count} integration test(s) skipped: {reason}")
    reasons = {}
    for report in other:
        reasons[_skip_reason(report)] = reasons.get(_skip_reason(report), 0) + 1
    for reason, count in sorted(reasons.items()):
        terminalreporter.write_line(f"{count} test(s) or module(s) skipped: {reason}")
