import os
import sys

# src/*.py call logging.basicConfig(filename="./log/stdout.log") at import time.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs(os.path.join(ROOT, "log"), exist_ok=True)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # noqa: E402

FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures")


@pytest.fixture
def fixture_text():
    def _load(name: str) -> str:
        with open(os.path.join(FIXTURE_DIR, name), "r", encoding="utf-8") as f:
            return f.read()
    return _load
