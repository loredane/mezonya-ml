"""Shared pytest configuration: put the project root on sys.path so tests
can import `scripts.*` and `api.*` without each file doing its own hack."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
