"""Common fakes used by unit tests.

This module centralizes lightweight stand-ins for:
- PyQt6.QtCore / PyQt6.QtGui / PyQt6.QtWidgets classes used by the app
- iartisanz app dependencies (EventBus, Database, SnackBar, etc.)

Compatibility shim.

Fakes were split into [tests/fakes/qt.py](tests/fakes/qt.py) and
[tests/fakes/app.py](tests/fakes/app.py). This module re-exports the original
names so existing tests keep working.
"""

from tests.fakes.app import (  # noqa: F401
    DirectoriesObject,
    FakeDatabase,
    FakeEventBus,
    FakeInitialSetupDialog,
    FakeMainWindow,
    FakeModule,
    FakeResource,
    FakeSnackBar,
    StoreBackedQSettings,
    fake_files,
    install_application_deps_fakes,
    install_main_window_deps_fakes,
)
from tests.fakes.qt import (  # noqa: F401
    FakeHBoxLayout,
    FakeLayoutItem,
    FakeQApplication,
    FakeQEasingCurve,
    FakeQFrame,
    FakeQMainWindow,
    FakeQPixmap,
    FakeQPoint,
    FakeQPropertyAnimation,
    FakeQSettings,
    FakeQSplashScreen,
    FakeQStatusBar,
    FakeQTimer,
    FakeQWidget,
    FakeSignal,
    FakeVBoxLayout,
    install_pyqt6_fakes,
)
