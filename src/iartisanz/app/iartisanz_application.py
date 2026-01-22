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


logger = logging.getLogger(__name__)


class BaseTorchAppApplication(QApplication):
    SPLASH_IMG = str(files("iartisanz.theme.images").joinpath("splash.webp"))

    def __init__(self, *args, **kwargs):
        myappid = "zcode.iartisanz.010"

        if platform.system() == "Windows":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        super().__init__(*args, **kwargs)

        style_data = files("iartisanz.theme").joinpath("stylesheet.qss").read_bytes()
        stylesheet = style_data.decode("utf-8")
        self.setStyleSheet(stylesheet)

        self.temp_path = tempfile.mkdtemp(prefix="iartisanz_")

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
        hide_nsfw = settings.value("hide_nsfw", True, type=bool)
        delete_lora_on_import = settings.value("delete_lora_on_import", False, type=bool)
        delete_model_on_import = settings.value("delete_model_on_import", False, type=bool)
        delete_model_after_conversion = settings.value("delete_model_after_conversion", False, type=bool)

        self.preferences = PreferencesObject(
            intermediate_images=intermediate_images,
            save_image_metadata=save_image_metadata,
            hide_nsfw=hide_nsfw,
            delete_lora_on_import=delete_lora_on_import,
            delete_model_on_import=delete_model_on_import,
            delete_model_after_conversion=delete_model_after_conversion,
        )

        data_path = settings.value("data_path", None, type=str)
        models_diffusers = settings.value("models_diffusers", None, type=str)
        models_singlefile = settings.value("models_singlefile", None, type=str)
        models_loras = settings.value("models_loras", None, type=str)
        outputs_images = settings.value("outputs_images", None, type=str)
        outputs_source_images = settings.value("outputs_source_images", None, type=str)
        outputs_source_masks = settings.value("outputs_source_masks", None, type=str)
        outputs_controlnet_source_images = settings.value("outputs_controlnet_source_images", None, type=str)
        outputs_conditioning_images = settings.value("outputs_conditioning_images", None, type=str)

        self.directories = DirectoriesObject(
            data_path=data_path,
            models_diffusers=models_diffusers,
            models_singlefile=models_singlefile,
            models_loras=models_loras,
            outputs_images=outputs_images,
            outputs_source_images=outputs_source_images,
            outputs_source_masks=outputs_source_masks,
            outputs_controlnet_source_images=outputs_controlnet_source_images,
            outputs_conditioning_images=outputs_conditioning_images,
            temp_path=self.temp_path,
        )

        if any(
            not v
            for v in [
                data_path,
                models_diffusers,
                models_singlefile,
                models_loras,
                outputs_images,
                outputs_source_images,
                outputs_source_masks,
                outputs_controlnet_source_images,
                outputs_conditioning_images,
            ]
        ):
            return False
        return True

    def cleanup_on_exit(self):
        try:
            from iartisanz.app.model_manager import get_model_manager

            get_model_manager().clear()
        except Exception as e:
            logger.debug("Failed to clear model manager on exit: %s", e)

        self.cleanup_temp_path()

    def cleanup_temp_path(self):
        if self.temp_path and os.path.exists(self.temp_path):  # check if it exists
            try:
                shutil.rmtree(self.temp_path)
                logger.info(f"Temporary directory '{self.temp_path}' cleaned up successfully.")
            except OSError as e:
                logger.error(f"Error cleaning up temporary directory: {e}")

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
