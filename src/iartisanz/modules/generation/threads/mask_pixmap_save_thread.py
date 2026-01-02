import os
from datetime import datetime

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPixmap


class MaskPixmapSaveThread(QThread):
    error = pyqtSignal(str)
    save_finished = pyqtSignal(str, str, str)

    def __init__(
        self,
        pixmap: QPixmap,
        opacity: float = 1.0,
        prefix: str = "img",
        temp_path: str = "tmp/",
        thumb_width: int = 80,
        thumb_height: int = 80,
    ):
        super().__init__()

        self.prefix = prefix
        self.temp_path = temp_path
        self.pixmap = pixmap
        self.opacity = opacity
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height

    def run(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        base_name = f"{self.prefix}_{timestamp}"

        mask_filename = f"{base_name}.png"
        mask_path = os.path.join(self.temp_path, mask_filename)
        self.pixmap.save(mask_path)

        mask_filename_final = f"{base_name}_final.png"
        mask_thumb_filename = f"{base_name}_thumb.png"
        mask_final_path = os.path.join(self.temp_path, mask_filename_final)
        mask_thumb_path = os.path.join(self.temp_path, mask_thumb_filename)

        white_background_pixmap = QPixmap(self.pixmap.size())
        white_background_pixmap.fill(Qt.GlobalColor.white)

        painter = QPainter(white_background_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        painter.setOpacity(self.opacity)

        painter.drawPixmap(0, 0, self.pixmap)
        painter.end()

        grayscale_image = white_background_pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
        grayscale_pixmap = QPixmap.fromImage(grayscale_image)

        grayscale_pixmap.save(mask_final_path)
        thumbnail_pixmap = grayscale_pixmap.scaled(self.thumb_width, self.thumb_height)
        thumbnail_pixmap.save(mask_thumb_path)

        self.save_finished.emit(mask_path, mask_final_path, mask_thumb_path)
