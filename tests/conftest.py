"""Pytest configuration.

When running under tools like `uv run`, the repository root may not be on
`sys.path` during test collection. Ensure both the project root and the `tests`
package are importable so tests can share helpers (e.g. `tests.fakes`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

for path in (str(PROJECT_ROOT), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication exists for PyQt6 widget tests.

    We force an offscreen platform so tests run headless under CI.
    Only tests that explicitly request this fixture will import real PyQt6.
    """

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def fake_superqt(monkeypatch, qapp):
    """Provide a minimal fake `superqt` module.

    The real `superqt.QLabeledDoubleSlider` can crash (abort) in some headless
    environments. For unit tests, we only need `setRange`, `setValue`, and
    `value`.
    """

    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QWidget

    class FakeQLabeledDoubleSlider(QWidget):
        valueChanged = pyqtSignal(float)
        sliderReleased = pyqtSignal()

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._min = 0.0
            self._max = 1.0
            self._value = 0.0

        def setRange(self, vmin: float, vmax: float):
            self._min = float(vmin)
            self._max = float(vmax)

        def setValue(self, v: float):
            v = float(v)
            if v < self._min:
                v = self._min
            if v > self._max:
                v = self._max
            self._value = v
            # Intentionally do not emit `valueChanged`.
            # LayerManagerWidget sets slider values during selection changes;
            # emitting here would trigger change handlers and can destabilize
            # headless test runs.

        def value(self) -> float:
            return float(self._value)

    class FakeQLabeledSlider(QWidget):
        valueChanged = pyqtSignal(int)
        sliderReleased = pyqtSignal()

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._min = 0
            self._max = 100
            self._value = 0

        def setRange(self, vmin: int, vmax: int):
            self._min = int(vmin)
            self._max = int(vmax)

        def setValue(self, v: int):
            v = int(v)
            if v < self._min:
                v = self._min
            if v > self._max:
                v = self._max
            self._value = v
            # Intentionally do not emit `valueChanged`.

        def value(self) -> int:
            return int(self._value)

    fake_superqt_mod = ModuleType("superqt")
    fake_superqt_mod.QLabeledDoubleSlider = FakeQLabeledDoubleSlider
    fake_superqt_mod.QLabeledSlider = FakeQLabeledSlider
    monkeypatch.setitem(sys.modules, "superqt", fake_superqt_mod)

    # If the widget module was imported before this fixture ran, it will have
    # already imported the real superqt slider class. Force a re-import so the
    # fake takes effect.
    sys.modules.pop("iartisanz.modules.generation.image.layer_manager_widget", None)
    sys.modules.pop("iartisanz.modules.generation.source_image.source_image_dialog", None)

    return fake_superqt_mod
