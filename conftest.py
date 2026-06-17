"""Ensure src/ is on sys.path for all test runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
