from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import QLabel


class ImageLabelWidget(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setScaledContents(True)

    def setPixmap(self, pixmap):
        if pixmap.isNull():
            super().setPixmap(pixmap)
        else:
            self.pixmap = pixmap
            self.update()

    def paintEvent(self, _):
        if hasattr(self, "pixmap") and isinstance(self.pixmap, QPixmap):
            painter = QPainter(self)
            pixmap = self.pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            point = int((self.width() - pixmap.width()) / 2), int((self.height() - pixmap.height()) / 2)
            painter.drawPixmap(QRect(*point, pixmap.width(), pixmap.height()), pixmap)
            painter.end()
