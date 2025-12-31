import importlib
import sys
from types import SimpleNamespace

from tests.fakes.app import FakeModule, install_main_window_deps_fakes
from tests.fakes.qt import FakeQSettings, FakeQWidget, install_pyqt6_fakes


def _install_fakes(monkeypatch, *, settings=None, modules=None):
    install_pyqt6_fakes(monkeypatch)
    install_main_window_deps_fakes(monkeypatch, modules=modules)

    # Seed QSettings values (optional)
    # Use a shared settings instance by patching the class to return pre-seeded stores.
    if settings is not None:
        original = FakeQSettings

        class _SeededQSettings(original):
            def __init__(self, org: str, app: str):
                super().__init__(org, app)
                self._store.update(settings)

        qtcore = sys.modules["PyQt6.QtCore"]
        qtcore.QSettings = _SeededQSettings


def _import_main_window_module(monkeypatch, *, settings=None, modules=None):
    _install_fakes(monkeypatch, settings=settings, modules=modules)
    mod = importlib.import_module("iartisanz.app.main_window")
    return importlib.reload(mod)


def test_init_initializes_database_and_loads_default_module(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()

    w = mainmod.MainWindow(dirs, prefs)

    assert w._title == "Image Artisan Z"
    assert w._min_size == (1300, 800)

    # Event bus subscriptions
    topics = [t for (t, _) in w.event_bus.subscriptions]
    assert "show_snackbar" in topics
    assert "status_message" in topics

    # Database initialized and tables created
    assert w.database.path == "/data/app.db"
    created = [name for (name, _) in w.database.tables]
    assert created == ["lora_model", "model"]

    # GUI defaults and module loading
    assert w.gui_options["left_menu_expanded"] is True
    assert w.gui_options["current_module"] == "Generation"
    assert isinstance(w.module, FakeModule)
    assert w.module.directories is dirs
    assert w.module.preferences is prefs
    assert w.window_loaded is True


def test_close_splash_sets_timer_finished_when_not_loaded(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()
    app = mainmod.QApplication.instance()

    w = mainmod.MainWindow(dirs, prefs)
    w.window_loaded = False
    w.timer_finished = False

    w.close_splash()

    assert w.timer_finished is True
    assert app.close_splash_calls == 0


def test_close_splash_calls_app_close_splash_when_loaded(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()
    app = mainmod.QApplication.instance()

    w = mainmod.MainWindow(dirs, prefs)
    w.window_loaded = True

    w.close_splash()

    assert app.close_splash_calls == 1


def test_show_snackbar_queues_and_displays_when_closed(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()

    w = mainmod.MainWindow(dirs, prefs)
    assert w.snackbar_closed is True

    w.show_snackbar("hello")

    assert w.snackbar.message == "hello"
    assert w.snackbar._shown == 1
    assert w.snackbar_animation.started == 1
    assert w.snackbar_closed is False
    assert w.snackbar_queue == []


def test_on_snackbar_animation_finished_starts_timer_then_hides(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()

    w = mainmod.MainWindow(dirs, prefs)

    # Start by showing one snackbar -> starts "show" animation
    w.show_snackbar("first")
    assert w.snackbar.message == "first"
    assert w.snackbar_animation.started == 1

    # When the show animation finishes, it should start the hide timer
    w.on_snackbar_animation_finished()
    assert w.snackbar_hide_animation is True
    assert w.snackbar_timer.started_with == w.snackbar_duration

    # Simulate timer timeout calling hide_snackbar -> starts "hide" animation
    w.hide_snackbar()
    assert w.snackbar_animation.started == 2

    # Ensure there is a next message queued before hide animation finishes
    w.snackbar_queue.append("next")

    # When the hide animation finishes, it should hide and immediately show next
    w.on_snackbar_animation_finished()
    assert w.snackbar._hidden == 1
    assert w.snackbar_closed is False
    assert w.snackbar.message == "next"
    assert w.snackbar_animation.started == 3


def test_load_module_replaces_existing_module_and_updates_label(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()

    w = mainmod.MainWindow(dirs, prefs)
    old = w.module
    w.workspace_layout.addWidget(old)

    class _NewModule(FakeQWidget):
        def __init__(self, directories, preferences):
            super().__init__()
            self.directories = directories
            self.preferences = preferences

    w.load_module(_NewModule, "NewLabel")

    assert old._closed == 1
    assert old._deleted_later == 1
    assert isinstance(w.module, _NewModule)
    assert w.gui_options["current_module"] == "NewLabel"
    assert w.workspace_layout.count() == 1
    assert w.workspace_layout.itemAt(0).widget() is w.module


def test_load_module_typeerror_shows_snackbar(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()
    w = mainmod.MainWindow(dirs, prefs)

    def _bad_module(_directories, _preferences):
        raise TypeError("bad module signature")

    w.snackbar_closed = True
    w.load_module(_bad_module, "Bad")

    assert w.snackbar.message == "bad module signature"
    assert w.snackbar._shown == 1


def test_on_show_snackbar_event_only_triggers_on_show(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()
    w = mainmod.MainWindow(dirs, prefs)

    w.on_show_snackbar_event({"action": "noop", "message": "x"})
    assert w.snackbar_queue == []

    w.on_show_snackbar_event({"action": "show", "message": "ok"})
    assert w.snackbar.message == "ok"


def test_on_status_message_event_changes_status_bar_message(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()
    w = mainmod.MainWindow(dirs, prefs)

    w.on_status_message_event({"action": "noop", "message": "x"})
    assert w.status_bar.last_message == "Ready"

    w.on_status_message_event({"action": "change", "message": "Working"})
    assert w.status_bar.last_message == "Working"


def test_close_event_saves_settings_closes_module_and_deletes_children(monkeypatch):
    mainmod = _import_main_window_module(monkeypatch, settings={})
    dirs = SimpleNamespace(data_path="/data")
    prefs = SimpleNamespace()
    w = mainmod.MainWindow(dirs, prefs)

    # prepare module and children
    w.module = FakeQWidget()
    child1 = FakeQWidget()
    child2 = FakeQWidget()
    w._children = [child1, child2]

    # ensure gui_options exists as expected
    w.gui_options["current_module"] = "Generation"

    w.closeEvent(event=object())

    # Settings saved under groups
    assert w.settings.value("geometry") is None  # outside group should not resolve
    w.settings.beginGroup("main_window")
    assert w.settings.value("geometry") == b"geom"
    assert w.settings.value("windowState") == b"state"
    w.settings.endGroup()

    w.settings.beginGroup("gui")
    assert w.settings.value("current_module") == "Generation"
    w.settings.endGroup()

    assert w.module._closed == 1
    assert child1._deleted_later == 1
    assert child2._deleted_later == 1
    assert w.super_close_event_called == 1
