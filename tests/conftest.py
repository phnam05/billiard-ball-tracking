import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: renders a synthetic clip and runs the full pipeline"
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
