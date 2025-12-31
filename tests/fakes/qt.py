"""Qt/PyQt6 fakes used by unit tests.

This module contains lightweight stand-ins for the subset of PyQt6 classes used
by the app and exercised by tests.

Install into `sys.modules` via `install_pyqt6_fakes()`.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple


class FakeSignal:
    def __init__(self):
        self._connected: list[Callable[..., Any]] = []
        self._emitted = 0

    def connect(self, fn: Callable[..., Any]):
        self._connected.append(fn)

    def emit(self, *args: Any, **kwargs: Any):
        self._emitted += 1
        for fn in list(self._connected):
            fn(*args, **kwargs)


class FakeQPoint:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class FakeQEasingCurve:
    class Type:
        InOutSine = "InOutSine"


class FakeQTimer:
    def __init__(self):
        self.timeout = FakeSignal()
        self.started_with: Optional[int] = None
        self.stopped = 0
        self.single_shots: list[Tuple[int, Callable[[], Any]]] = []

    def singleShot(self, ms: int, fn: Callable[[], Any]):
        self.single_shots.append((ms, fn))

    def start(self, ms: int):
        self.started_with = ms

    def stop(self):
        self.stopped += 1


class FakeQPropertyAnimation:
    def __init__(self, target: Any, prop: bytes):
        self.target = target
        self.prop = prop
        self._easing = None
        self._duration = None
        self._start_value = None
        self._end_value = None
        self.started = 0
        self.finished = FakeSignal()

    def setEasingCurve(self, easing: Any):
        self._easing = easing

    def setDuration(self, duration: int):
        self._duration = duration

    def setStartValue(self, v: Any):
        self._start_value = v

    def setEndValue(self, v: Any):
        self._end_value = v

    def start(self):
        self.started += 1


class FakeQSettings:
    """Instance-backed store (per test), used by MainWindow tests."""

    def __init__(self, org: str, app: str):
        self.org = org
        self.app = app
        self._group_stack: list[str] = []
        self._store: Dict[str, Any] = {}

    def beginGroup(self, group: str):
        self._group_stack.append(group)

    def endGroup(self):
        if self._group_stack:
            self._group_stack.pop()

    def _prefix(self) -> str:
        return "/".join(self._group_stack) + ("/" if self._group_stack else "")

    def value(self, key: str, default: Any = None, type: Any = None):
        k = f"{self._prefix()}{key}"
        v = self._store.get(k, default)
        if type is bool:
            return bool(v)
        if type is str and v is not None:
            return str(v)
        return v

    def setValue(self, key: str, value: Any):
        k = f"{self._prefix()}{key}"
        self._store[k] = value


class FakeQApplication:
    _instance = None

    def __init__(self, *args: Any, **kwargs: Any):
        FakeQApplication._instance = self
        self.close_splash_calls = 0
        self.aboutToQuit = FakeSignal()
        self._stylesheet: Optional[str] = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = FakeQApplication()
        return cls._instance

    def close_splash(self):
        self.close_splash_calls += 1

    def setStyleSheet(self, stylesheet: str):
        self._stylesheet = stylesheet


class FakeQWidget:
    def __init__(self, *args: Any, **kwargs: Any):
        self._deleted_later = 0
        self._closed = 0
        self._layout = None

    def setLayout(self, layout: Any):
        self._layout = layout

    def layout(self):
        return self._layout

    def deleteLater(self):
        self._deleted_later += 1

    def close(self):
        self._closed += 1


class FakeLayoutItem:
    def __init__(self, widget: Any):
        self._widget = widget

    def widget(self):
        return self._widget


class FakeVBoxLayout:
    def __init__(self):
        self._widgets: list[Any] = []
        self._layouts: list[Any] = []
        self._margins = None
        self._spacing = None

    def setContentsMargins(self, *args: Any):
        self._margins = args

    def setSpacing(self, s: int):
        self._spacing = s

    def addWidget(self, w: Any):
        self._widgets.append(w)

    def addLayout(self, layout: Any):
        self._layouts.append(layout)

    def count(self) -> int:
        return len(self._widgets)

    def itemAt(self, idx: int):
        return FakeLayoutItem(self._widgets[idx])

    def removeWidget(self, w: Any):
        self._widgets = [x for x in self._widgets if x is not w]


class FakeHBoxLayout(FakeVBoxLayout):
    def __init__(self):
        super().__init__()
        self._stretches: Dict[int, int] = {}

    def setStretch(self, idx: int, stretch: int):
        self._stretches[idx] = stretch


class FakeQFrame(FakeQWidget):
    class Shape:
        Box = 0x01

    class Shadow:
        Sunken = 0x02

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.object_name = None
        self.frame_style = None
        self.layout = None

    def setObjectName(self, name: str):
        self.object_name = name

    def setFrameStyle(self, style: Any):
        self.frame_style = style

    def setLayout(self, layout: Any):
        self.layout = layout


class FakeQStatusBar(FakeQWidget):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.last_message: Optional[str] = None

    def showMessage(self, msg: str):
        self.last_message = msg


class FakeQMainWindow(FakeQWidget):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._title = None
        self._min_size = None
        self._central_widget = None
        self._status_bar = None
        self._geometry_restored = None
        self._state_restored = None
        self._saved_geometry = b"geom"
        self._saved_state = b"state"
        self._children: list[Any] = []
        self.super_close_event_called = 0
        self._width = 800
        self._height = 600

    def setWindowTitle(self, title: str):
        self._title = title

    def setMinimumSize(self, w: int, h: int):
        self._min_size = (w, h)

    def restoreGeometry(self, geometry: Any):
        self._geometry_restored = geometry

    def restoreState(self, state: Any):
        self._state_restored = state

    def setStatusBar(self, sb: Any):
        self._status_bar = sb

    def setCentralWidget(self, w: Any):
        self._central_widget = w

    def saveGeometry(self):
        return self._saved_geometry

    def saveState(self):
        return self._saved_state

    def findChildren(self, _cls: Any):
        return list(self._children)

    def closeEvent(self, event: Any):
        self.super_close_event_called += 1

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class FakeQPixmap:
    def __init__(self, path: str):
        self.path = path


class FakeQSplashScreen:
    def __init__(self, pixmap: Any):
        self.pixmap = pixmap
        self.messages: list[Tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.show_called = 0
        self.closed = 0

    def showMessage(self, *args: Any, **kwargs: Any):
        self.messages.append((args, kwargs))

    def show(self):
        self.show_called += 1

    def close(self):
        self.closed += 1


def install_pyqt6_fakes(
    monkeypatch,
    *,
    include_gui: bool = False,
    include_splash: bool = False,
    qtcore_overrides: Optional[dict[str, Any]] = None,
):
    """Install PyQt6 module fakes into sys.modules."""

    pyqt6 = ModuleType("PyQt6")
    qtcore = ModuleType("PyQt6.QtCore")
    qtwidgets = ModuleType("PyQt6.QtWidgets")

    qtcore.QEasingCurve = FakeQEasingCurve
    qtcore.QPoint = FakeQPoint
    qtcore.QPropertyAnimation = FakeQPropertyAnimation
    qtcore.QSettings = FakeQSettings
    qtcore.QTimer = FakeQTimer

    if qtcore_overrides:
        for k, v in qtcore_overrides.items():
            setattr(qtcore, k, v)

    qtwidgets.QApplication = FakeQApplication
    qtwidgets.QFrame = FakeQFrame
    qtwidgets.QHBoxLayout = FakeHBoxLayout
    qtwidgets.QMainWindow = FakeQMainWindow
    qtwidgets.QStatusBar = FakeQStatusBar
    qtwidgets.QVBoxLayout = FakeVBoxLayout
    qtwidgets.QWidget = FakeQWidget

    monkeypatch.setitem(sys.modules, "PyQt6", pyqt6)
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", qtcore)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qtwidgets)

    if include_gui or include_splash:
        qtgui = ModuleType("PyQt6.QtGui")
        qtgui.QPixmap = FakeQPixmap
        monkeypatch.setitem(sys.modules, "PyQt6.QtGui", qtgui)

    if include_splash:
        qtwidgets.QSplashScreen = FakeQSplashScreen
        monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qtwidgets)

    # Keep tests isolated: other tests may have instantiated a subclass of
    # QApplication which would otherwise persist as the global instance.
    FakeQApplication._instance = None
