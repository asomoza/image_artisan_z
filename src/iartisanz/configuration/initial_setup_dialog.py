import os

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QDialog, QFileDialog, QLabel, QPushButton, QVBoxLayout

from iartisanz.app.directories import DirectoriesObject
from iartisanz.app.preferences import PreferencesObject


class InitialSetupDialog(QDialog):
    border_color = QColor("#ff6b6b6b")

    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dialog_width = 650
        self.dialog_height = 720

        self.directories = directories
        self.preferences = preferences

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        select_directory_button = QPushButton("Select the image outputs directory")
        select_directory_button.clicked.connect(self.select_directory)
        main_layout.addWidget(select_directory_button)

        defaults_button = QPushButton("Use defaults")
        defaults_button.clicked.connect(self.on_defaults)
        main_layout.addWidget(defaults_button)

        self.setLayout(main_layout)

    def finish_setup(self):
        self.close()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(0, 0, self.width(), 0)
        painter.drawLine(self.width(), 0, self.width(), self.height())
        painter.drawLine(0, self.height(), self.width(), self.height())

    def select_directory(self):
        home_dir = os.path.expanduser("~")

        dialog = QFileDialog()
        options = (
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontUseNativeDialog
            | QFileDialog.Option.ReadOnly
            | QFileDialog.Option.HideNameFilterDetails
        )
        dialog.setOptions(options)

        settings = QSettings("ZCode", "ImageArtisanZ")

        selected_path = dialog.getExistingDirectory(None, "Select a directory", home_dir)

        self.directories.outputs_images = selected_path
        settings.setValue("outputs_images", selected_path)

        sender_button = self.sender()
        sender_button.parent_widget.directory_label.setText(selected_path)
        self.close()

    def on_defaults(self):
        base_dirs = ["Documents", "Image Artisan Z"]

        sub_dirs = {
            "outputs_images": os.path.join("outputs", "images"),
        }

        home_dir = os.path.expanduser("~")

        for directory in base_dirs:
            home_dir = os.path.join(home_dir, directory)
            if not os.path.exists(home_dir):
                os.makedirs(home_dir)

        for key, directory in sub_dirs.items():
            sub_dir = os.path.join(home_dir, directory)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            setattr(self.directories, key, sub_dir)

        settings = QSettings("ZCode", "ImageArtisanZ")
        settings.setValue("outputs_images", self.directories.outputs_images)
        self.close()
