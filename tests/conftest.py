"""Pytest configuration.

When running under tools like `uv run`, the repository root may not be on
`sys.path` during test collection. Ensure both the project root and the `tests`
package are importable so tests can share helpers (e.g. `tests.fakes`).
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

for path in (str(PROJECT_ROOT), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
