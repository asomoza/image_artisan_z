import importlib
import os
import tempfile

from tests.fakes.app import FakeInitialSetupDialog, FakeMainWindow, install_application_deps_fakes
from tests.fakes.qt import FakeQPixmap, FakeQSplashScreen


def _install_fakes(monkeypatch, *, settings_store=None):
    install_application_deps_fakes(monkeypatch, settings_store=settings_store)


def _import_app_module(monkeypatch, *, settings_store=None):
    _install_fakes(monkeypatch, settings_store=settings_store)
    mod = importlib.import_module("iartisanz.app.iartisanz_application")
    return importlib.reload(mod)


def test_check_initial_setup_returns_false_when_any_required_directory_is_missing(monkeypatch):
    settings = {
        # preferences
        "intermediate_images": True,
        "save_image_metadata": False,
        "hide_nsfw": True,
        "delete_lora_on_import": True,
        "delete_model_on_import": False,
        "delete_model_after_conversion": True,
        # directories: missing outputs_source_images
        "data_path": "/data",
        "models_diffusers": "/models/diffusers",
        "models_singlefile": "/models/singlefile",
        "models_loras": "/models/loras",
        "outputs_images": "/outputs/images",
        "outputs_source_images": None,
    }
    appmod = _import_app_module(monkeypatch, settings_store=settings)

    app = appmod.BaseTorchAppApplication.__new__(appmod.BaseTorchAppApplication)
    app.temp_path = "/tmp/iartisanz_test"

    ok = appmod.BaseTorchAppApplication.check_initial_setup(app)

    assert ok is False
    assert app.preferences.intermediate_images is True
    assert app.preferences.save_image_metadata is False
    assert app.preferences.hide_nsfw is True
    assert app.preferences.delete_lora_on_import is True
    assert app.preferences.delete_model_on_import is False
    assert app.preferences.delete_model_after_conversion is True

    assert app.directories.data_path == "/data"
    assert app.directories.models_diffusers == "/models/diffusers"
    assert app.directories.models_singlefile == "/models/singlefile"
    assert app.directories.models_loras == "/models/loras"
    assert app.directories.outputs_images == "/outputs/images"
    assert app.directories.outputs_source_images is None
    assert app.directories.temp_path == "/tmp/iartisanz_test"


def test_check_initial_setup_returns_true_when_all_required_directories_are_set(monkeypatch):
    settings = {
        "data_path": "/data",
        "models_diffusers": "/models/diffusers",
        "models_singlefile": "/models/singlefile",
        "models_loras": "/models/loras",
        "models_controlnets": "/models/controlnet",
        "outputs_images": "/outputs/images",
        "outputs_source_images": "/outputs/source",
        "outputs_source_masks": "/outputs/source_masks",
        "outputs_controlnet_source_images": "/outputs/controlnet_source_images",
        "outputs_conditioning_images": "/outputs/conditioning_images",
    }
    appmod = _import_app_module(monkeypatch, settings_store=settings)

    app = appmod.BaseTorchAppApplication.__new__(appmod.BaseTorchAppApplication)
    app.temp_path = "/tmp/iartisanz_test"

    ok = appmod.BaseTorchAppApplication.check_initial_setup(app)

    assert ok is True
    assert app.directories.outputs_source_images == "/outputs/source"
    assert app.directories.temp_path == "/tmp/iartisanz_test"


def test_cleanup_temp_path_removes_existing_temp_directory(monkeypatch):
    appmod = _import_app_module(monkeypatch, settings_store={})

    tmpdir = tempfile.mkdtemp(prefix="iartisanz_test_")
    with open(os.path.join(tmpdir, "file.txt"), "w", encoding="utf-8") as f:
        f.write("x")

    app = appmod.BaseTorchAppApplication.__new__(appmod.BaseTorchAppApplication)
    app.temp_path = tmpdir

    appmod.BaseTorchAppApplication.cleanup_temp_path(app)

    assert not os.path.exists(tmpdir)


def test_cleanup_temp_path_is_noop_when_temp_path_does_not_exist(monkeypatch):
    appmod = _import_app_module(monkeypatch, settings_store={})

    app = appmod.BaseTorchAppApplication.__new__(appmod.BaseTorchAppApplication)
    app.temp_path = os.path.join(tempfile.gettempdir(), "iartisanz_does_not_exist_12345")

    # Should not raise
    appmod.BaseTorchAppApplication.cleanup_temp_path(app)


def test_close_splash_closes_splash_and_shows_window(monkeypatch):
    appmod = _import_app_module(monkeypatch, settings_store={})

    app = appmod.BaseTorchAppApplication.__new__(appmod.BaseTorchAppApplication)
    app.splash = FakeQSplashScreen(FakeQPixmap("x"))
    app.window = FakeMainWindow("dirs", "prefs")

    appmod.BaseTorchAppApplication.close_splash(app)

    assert app.splash.closed == 1
    assert app.window.shown == 1


def test_close_splash_is_noop_without_splash(monkeypatch):
    appmod = _import_app_module(monkeypatch, settings_store={})

    app = appmod.BaseTorchAppApplication.__new__(appmod.BaseTorchAppApplication)
    app.splash = None
    app.window = FakeMainWindow("dirs", "prefs")

    appmod.BaseTorchAppApplication.close_splash(app)

    assert app.window.shown == 0


def test_load_main_window_creates_splash_sets_message_and_creates_main_window(monkeypatch):
    appmod = _import_app_module(monkeypatch, settings_store={})

    app = appmod.BaseTorchAppApplication.__new__(appmod.BaseTorchAppApplication)
    app.directories = "DIRS"
    app.preferences = "PREFS"

    appmod.BaseTorchAppApplication.load_main_window(app)

    assert isinstance(app.splash, FakeQSplashScreen)
    assert isinstance(app.splash.pixmap, FakeQPixmap)
    assert app.splash.pixmap.path == appmod.BaseTorchAppApplication.SPLASH_IMG
    assert app.splash.show_called == 1

    # Message content and key kwargs
    assert app.splash.messages, "Expected showMessage to be called"
    (args, kwargs) = app.splash.messages[-1]
    assert args[0] == "Loading..."
    assert kwargs["alignment"] == appmod.Qt.AlignmentFlag.AlignBottom
    assert kwargs["color"] == appmod.Qt.GlobalColor.white

    assert isinstance(app.window, FakeMainWindow)
    assert app.window.directories == "DIRS"
    assert app.window.preferences == "PREFS"


def test_init_runs_initial_setup_dialog_when_check_initial_setup_is_false(monkeypatch):
    appmod = _import_app_module(monkeypatch, settings_store={})

    monkeypatch.setattr(appmod.platform, "system", lambda: "Linux", raising=True)
    monkeypatch.setattr(appmod.tempfile, "mkdtemp", lambda prefix: "/tmp/iartisanz_init_test", raising=True)

    def _fake_check(self):
        # Ensure __init__ passed temp_path creation before calling this.
        assert self.temp_path == "/tmp/iartisanz_init_test"
        # Provide objects used by InitialSetupDialog(self.directories, self.preferences)
        self.directories = "DIRS_OBJ"
        self.preferences = "PREFS_OBJ"
        return False

    def _fake_load(self):
        self.loaded_main = True

    monkeypatch.setattr(appmod.BaseTorchAppApplication, "check_initial_setup", _fake_check, raising=True)
    monkeypatch.setattr(appmod.BaseTorchAppApplication, "load_main_window", _fake_load, raising=True)

    app = appmod.BaseTorchAppApplication([])

    assert app.temp_path == "/tmp/iartisanz_init_test"
    assert getattr(app, "loaded_main", False) is True
    assert isinstance(app.dialog, FakeInitialSetupDialog)
    assert app.dialog.directories == "DIRS_OBJ"
    assert app.dialog.preferences == "PREFS_OBJ"
    assert app.dialog.executed == 1
    assert app.cleanup_on_exit in app.aboutToQuit._connected
    assert isinstance(app._stylesheet, str)
    assert "qss" in app._stylesheet


def test_init_does_not_run_initial_setup_dialog_when_check_initial_setup_is_true(monkeypatch):
    appmod = _import_app_module(monkeypatch, settings_store={})

    monkeypatch.setattr(appmod.platform, "system", lambda: "Linux", raising=True)
    monkeypatch.setattr(appmod.tempfile, "mkdtemp", lambda prefix: "/tmp/iartisanz_init_test2", raising=True)

    def _fake_check(self):
        self.directories = "DIRS_OBJ"
        self.preferences = "PREFS_OBJ"
        return True

    def _fake_load(self):
        self.loaded_main = True

    class _FailDialog:
        def __init__(self, *args, **kwargs):
            raise AssertionError("InitialSetupDialog should not be created when initial setup is complete")

    monkeypatch.setattr(appmod.BaseTorchAppApplication, "check_initial_setup", _fake_check, raising=True)
    monkeypatch.setattr(appmod, "InitialSetupDialog", _FailDialog, raising=True)
    monkeypatch.setattr(appmod.BaseTorchAppApplication, "load_main_window", _fake_load, raising=True)

    app = appmod.BaseTorchAppApplication([])

    assert app.temp_path == "/tmp/iartisanz_init_test2"
    assert getattr(app, "loaded_main", False) is True
    assert not hasattr(app, "dialog")
