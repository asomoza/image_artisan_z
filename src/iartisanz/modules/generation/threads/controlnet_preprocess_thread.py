from PyQt6.QtCore import QThread, pyqtSignal

from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer


class ControlnetPreprocessThread(QThread):
    error = pyqtSignal(str)
    preprocessor_finished = pyqtSignal(str)

    def __init__(
        self,
        layer: ImageEditorLayer,
    ):
        super().__init__()

        self.layer = layer

        self.preprocessor = None
        self.preprocessor_model = None

    def run(self): ...
