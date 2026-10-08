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


