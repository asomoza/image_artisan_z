"""App-level fakes used by unit tests.

This module contains fakes for non-Qt application dependencies and installers
that patch `sys.modules` so app modules can be imported without the real stack.
"""

from __future__ import annotations

import importlib.resources
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Optional

from tests.fakes.qt import (
    FakeQApplication,
    FakeQWidget,
    install_pyqt6_fakes,
)


class FakeSnackBar(FakeQWidget):
    def __init__(self, parent: Any = None):
        super().__init__()
        from tests.fakes.qt import FakeSignal

        self.parent = parent
        self.closed = FakeSignal()
        self.message: Optional[str] = None
        self._shown = 0
        self._hidden = 0
        self._width = 200
        self._height = 50

    def show(self):
        self._shown += 1

    def hide(self):
        self._hidden += 1

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class FakeEventBus:
    def __init__(self):
        self.subscriptions: list[tuple[str, Callable[..., Any]]] = []

    def subscribe(self, topic: str, handler: Callable[..., Any]):
        self.subscriptions.append((topic, handler))


class FakeDatabase:
    def __init__(self, path: str):
        self.path = path
        self.tables: list[tuple[str, list[Any]]] = []

    def create_table(self, name: str, columns: Iterable[Any]):
        self.tables.append((name, list(columns)))


class FakeModule(FakeQWidget):
    def __init__(self, directories: Any, preferences: Any):
        super().__init__()
        self.directories = directories
        self.preferences = preferences


class FakeMainWindow:
    def __init__(self, directories: Any, preferences: Any):
        self.directories = directories
        self.preferences = preferences
        self.shown = 0

    def show(self):
        self.shown += 1


class FakeInitialSetupDialog:
    def __init__(self, directories: Any, preferences: Any):
        self.directories = directories
        self.preferences = preferences
        self.executed = 0

    def exec(self) -> int:
        self.executed += 1
        return 0


@dataclass
class DirectoriesObject:
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class PreferencesObject:
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeResource:
    def __init__(self, parts: Iterable[str]):
        self._parts = tuple(parts)

    def joinpath(self, name: str):
        return FakeResource(self._parts + (name,))

    def read_bytes(self) -> bytes:
        return b"/* qss */"

    def __str__(self) -> str:
        return "/".join(self._parts)


def fake_files(package: str):
    return FakeResource((package,))


def install_main_window_deps_fakes(monkeypatch, *, modules: Optional[dict[str, Any]] = None):
    """Install app fakes needed by iartisanz.app.main_window."""

    mod_dirs = ModuleType("iartisanz.app.directories")
    mod_dirs.DirectoriesObject = object
    monkeypatch.setitem(sys.modules, "iartisanz.app.directories", mod_dirs)

    mod_prefs = ModuleType("iartisanz.app.preferences")
    mod_prefs.PreferencesObject = object
    monkeypatch.setitem(sys.modules, "iartisanz.app.preferences", mod_prefs)

    mod_bus = ModuleType("iartisanz.app.event_bus")
    mod_bus.EventBus = FakeEventBus
    monkeypatch.setitem(sys.modules, "iartisanz.app.event_bus", mod_bus)

    mod_snackbar = ModuleType("iartisanz.app.snackbar")
    mod_snackbar.SnackBar = FakeSnackBar
    monkeypatch.setitem(sys.modules, "iartisanz.app.snackbar", mod_snackbar)

    mod_db = ModuleType("iartisanz.utils.database")
    mod_db.Database = FakeDatabase
    monkeypatch.setitem(sys.modules, "iartisanz.utils.database", mod_db)

    mod_modules = ModuleType("iartisanz.app.modules")
    mod_modules.MODULES = modules or {"Generation": ("Generation", FakeModule)}
    monkeypatch.setitem(sys.modules, "iartisanz.app.modules", mod_modules)

    # Fake iartisanz.app.app (global accessors added by component registry)
    mod_app_app = ModuleType("iartisanz.app.app")
    mod_app_app.set_app_database_path = lambda path: None
    mod_app_app.set_app_directories = lambda dirs: None
    monkeypatch.setitem(sys.modules, "iartisanz.app.app", mod_app_app)

    # Fake iartisanz.app.migration (called lazily inside init_database)
    mod_migration = ModuleType("iartisanz.app.migration")
    mod_migration.run_migrations = lambda db, dirs: None
    monkeypatch.setitem(sys.modules, "iartisanz.app.migration", mod_migration)

    # Avoid importing real torch in MainWindow.__init__
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))


class StoreBackedQSettings:
    """Class-backed store (shared dict), used by application tests."""

    store: Dict[str, Any] = {}

    def __init__(self, org: str, app: str):
        self.org = org
        self.app = app

    def value(self, key: str, default: Any = None, type: Any = None):
        v = self.store.get(key, default)
        if type is bool:
            return bool(v)
        if type is str and v is not None:
            return str(v)
        return v


def install_application_deps_fakes(monkeypatch, *, settings_store: Optional[Dict[str, Any]] = None):
    """Install fakes needed by iartisanz.app.iartisanz_application."""

    qtcore_overrides = {
        "QSettings": StoreBackedQSettings,
        "Qt": SimpleNamespace(
            AlignmentFlag=SimpleNamespace(AlignBottom="AlignBottom"),
            GlobalColor=SimpleNamespace(white="white"),
        ),
    }

    install_pyqt6_fakes(monkeypatch, include_gui=True, include_splash=True, qtcore_overrides=qtcore_overrides)

    # Seed QSettings store
    StoreBackedQSettings.store = dict(settings_store or {})

    mod_dirs = ModuleType("iartisanz.app.directories")
    mod_dirs.DirectoriesObject = DirectoriesObject
    monkeypatch.setitem(sys.modules, "iartisanz.app.directories", mod_dirs)

    mod_prefs = ModuleType("iartisanz.app.preferences")
    mod_prefs.PreferencesObject = PreferencesObject
    monkeypatch.setitem(sys.modules, "iartisanz.app.preferences", mod_prefs)

    mod_main = ModuleType("iartisanz.app.main_window")
    mod_main.MainWindow = FakeMainWindow
    monkeypatch.setitem(sys.modules, "iartisanz.app.main_window", mod_main)

    mod_setup = ModuleType("iartisanz.configuration.initial_setup_dialog")
    mod_setup.InitialSetupDialog = FakeInitialSetupDialog
    monkeypatch.setitem(sys.modules, "iartisanz.configuration.initial_setup_dialog", mod_setup)

    # Ensure stylesheet read uses a safe fake resource
    monkeypatch.setattr(importlib.resources, "files", fake_files, raising=True)

    # Ensure the QApplication singleton doesn't leak between test modules
    FakeQApplication._instance = None
