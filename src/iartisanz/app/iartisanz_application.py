import ctypes
import logging
import os
import platform
import shutil
import tempfile
from importlib.resources import files

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from iartisanz.app.directories import DirectoriesObject
from iartisanz.app.main_window import MainWindow
from iartisanz.app.preferences import PreferencesObject
from iartisanz.configuration.initial_setup_dialog import InitialSetupDialog


class BaseTorchAppApplication(QApplication):
    SPLASH_IMG = str(files("iartisanz.theme.images").joinpath("splash.webp"))

    def __init__(self, *args, **kwargs):
        myappid = "zcode.iartisanz.010"

        if platform.system() == "Windows":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(__name__)

        style_data = files("iartisanz.theme").joinpath("stylesheet.qss").read_bytes()
        stylesheet = style_data.decode("utf-8")
        self.setStyleSheet(stylesheet)

        self.temp_dir = tempfile.mkdtemp(prefix="iravision_")

        self.window = None
        self.splash = None

        self.directories = None
        self.preferences = None

        self.aboutToQuit.connect(self.cleanup_on_exit)

        if not self.check_initial_setup():
            self.dialog = InitialSetupDialog(self.directories, self.preferences)
            self.dialog.exec()

        self.load_main_window()

    def check_initial_setup(self):
        settings = QSettings("ZCode", "ImageArtisanZ")

        intermediate_images = settings.value("intermediate_images", False, type=bool)
        save_image_metadata = settings.value("save_image_metadata", False, type=bool)

        self.preferences = PreferencesObject(
            intermediate_images=intermediate_images,
            save_image_metadata=save_image_metadata,
        )

        outputs_images = settings.value("outputs_images", None, type=str)

        self.directories = DirectoriesObject(
            outputs_images=outputs_images,
        )

        if any(
            not v
            for v in [
                outputs_images,
            ]
        ):
            return False
        return True

    def cleanup_on_exit(self):
        self.cleanup_temp_directory()

    def cleanup_temp_directory(self):
        if self.temp_dir and os.path.exists(self.temp_dir):  # check if it exists
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Temporary directory '{self.temp_dir}' cleaned up successfully.")
            except OSError as e:
                print(f"Error cleaning up temporary directory: {e}")

    def close_splash(self):
        if self.splash:
            self.splash.close()
            self.window.show()

    def load_main_window(self):
        splash_pix = QPixmap(self.SPLASH_IMG)
        self.splash = QSplashScreen(splash_pix)
        self.splash.showMessage(
            "Loading...",
            alignment=Qt.AlignmentFlag.AlignBottom,
            color=Qt.GlobalColor.white,
        )

        self.splash.show()

        self.window = MainWindow(self.directories, self.preferences)
