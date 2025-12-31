from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import (
    QAction,
    QContextMenuEvent,
    QImageWriter,
    QKeySequence,
    QMouseEvent,
    QScreen,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QGraphicsView, QMenu

from iartisanz.app.event_bus import EventBus
from iartisanz.modules.generation.dialogs.full_screen_preview import FullScreenPreview


if TYPE_CHECKING:
    from iartisanz.app.directories import DirectoriesObject
    from iartisanz.app.preferences import PreferencesObject


class ImageViewerSimpleWidget(QGraphicsView):
    def __init__(self, directories: DirectoriesObject, preferences: PreferencesObject):
        super().__init__()

        self.setAcceptDrops(True)

        self.directories = directories
        self.preferences = preferences
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.setScene(QGraphicsScene())

        self.selected_screen = None
        self.full_screen_preview = None

        self.context_menu = QMenu(self)
        self.save_action = self.context_menu.addAction("Save image")
        self.save_action.triggered.connect(self.save_image)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_image)

        self.initial_scale_factor = None
        self.json_graph = None
        self.pixmap_item = None

        self.moving = False
        self.last_mouse_position = None
        self.setMouseTracking(True)
        self.current_scale_factor = 1.0
        self.total_translation = QPointF(0, 0)
        self.min_scale_factor = 0.1
        self.max_scale_factor = 20.0

        self.event_bus = EventBus()

    def set_pixmap(self, pixmap):
        self.scene().clear()
        self.pixmap_item = self.scene().addPixmap(pixmap)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.initial_scale_factor = self.transform().m11()
        self.current_scale_factor = self.initial_scale_factor
        self.total_translation = QPointF(0, 0)
        if self.selected_screen is not None and self.full_screen_preview is not None:
            self.full_screen_preview.image_preview_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item is not None:
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self.moving:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.moving = True

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.moving = False

        super().keyReleaseEvent(event)

    def set_zoom_limits(self, min_scale: float, max_scale: float):
        self.min_scale_factor = min_scale
        self.max_scale_factor = max_scale

    def wheelEvent(self, event: QWheelEvent):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        zoomFactor = zoomInFactor if event.angleDelta().y() > 0 else zoomOutFactor
        target_scale = max(
            self.min_scale_factor,
            min(self.max_scale_factor, self.current_scale_factor * zoomFactor),
        )
        effective_factor = target_scale / self.current_scale_factor
        if effective_factor != 1.0:
            self.scale(effective_factor, effective_factor)
            self.current_scale_factor = target_scale

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.moving:
                self.last_mouse_position = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.moving:
                self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                delta = self.mapToScene(self.last_mouse_position) - self.mapToScene(event.pos())
                self.setSceneRect(self.sceneRect().translated(delta.x(), delta.y()))
                self.translate(delta.x() / self.current_scale_factor, delta.y() / self.current_scale_factor)
                self.total_translation += delta
                self.last_mouse_position = event.pos()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.moving:
                self.setCursor(Qt.CursorShape.OpenHandCursor)

        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent):
        menu = QMenu(self)
        menu.addAction(self.save_action)
        full_screen_preview_action: QAction | None = menu.addAction("Full Screen Preview")

        screens = QApplication.screens()
        submenu = QMenu("Submenu", self)

        none_action = submenu.addAction("None")
        none_action.setCheckable(True)
        none_action.setChecked(self.selected_screen is None)
        none_action.triggered.connect(lambda checked: self.on_monitor_selected(None))

        current_screen = self.window().screen()

        for i, screen in enumerate(screens, start=1):
            action = submenu.addAction(f"Monitor {i} - {screen.name()}")
            action.setCheckable(True)
            action.setChecked(screen == self.selected_screen)

            if screen == current_screen:
                action.setEnabled(False)

            action.triggered.connect(lambda checked, s=screen: self.on_monitor_selected(s))

        full_screen_preview_action.setMenu(submenu)
        menu.exec(event.globalPos())

    def on_monitor_selected(self, screen: Optional[QScreen]):
        self.selected_screen = screen

        if screen is not None:
            self.full_screen_preview = FullScreenPreview()
            self.full_screen_preview.move(screen.geometry().x(), screen.geometry().y())
            self.full_screen_preview.showFullScreen()
        else:
            self.full_screen_preview = None

    def set_json_graph(self, json_graph: str):
        self.json_graph = json_graph

    def _copy_source_image_and_rewrite_graph(self, json_graph: str, timestamp: str) -> str:
        try:
            data = json.loads(json_graph)
        except Exception:
            return json_graph

        nodes = data.get("nodes")
        if not isinstance(nodes, list):
            return json_graph

        updated = False
        dest_dir = Path(self.directories.outputs_source_images)

        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("class") != "ImageLoadNode" or node.get("name") != "source_image":
                continue

            state = node.get("state")
            if not isinstance(state, dict):
                continue

            src_path = state.get("path")
            if not isinstance(src_path, str) or not src_path.strip():
                continue

            src = Path(src_path)
            if not src.exists() or not src.is_file():
                continue

            ext = src.suffix if src.suffix else ".png"
            base_name = f"{timestamp}_source_image{ext}"
            dest = dest_dir / base_name

            # Avoid overwriting if the same timestamp is reused.
            if dest.exists():
                for i in range(1, 10_000):
                    candidate = dest_dir / f"{timestamp}_source_image_{i}{ext}"
                    if not candidate.exists():
                        dest = candidate
                        break

            try:
                shutil.copy2(src, dest)
            except Exception:
                # Copy failed; keep original path/JSON.
                continue

            state["path"] = str(dest)
            node["state"] = state
            updated = True

        if not updated:
            return json_graph

        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return json_graph

    def save_image(self):
        if self.pixmap_item is None:
            self.event_bus.publish("show_snackbar", {"action": "show", "message": "No image to save"})
            return

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            f"{self.directories.outputs_images}/{timestamp}.png",
            "PNG Images (*.png)",
        )
        if not output_path:
            return  # user cancelled

        if not output_path.lower().endswith(".png"):
            output_path += ".png"

        image = self.pixmap_item.pixmap().toImage()

        writer = QImageWriter(output_path, b"png")

        if self.preferences.save_image_metadata and self.json_graph:
            json_graph_to_save = self._copy_source_image_and_rewrite_graph(self.json_graph, timestamp)
            writer.setText("iartisanz_json_graph", json_graph_to_save)

        if not writer.write(image):
            self.event_bus.publish(
                "show_snackbar",
                {"action": "show", "message": f"Failed to save image: {writer.errorString()}"},
            )

    def dragMoveEvent(self, event):
        pass

    def dragEnterEvent(self, event):
        pass

    def dropEvent(self, event):
        pass

    def reset_zoom(self):
        if self.pixmap_item is not None:
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.current_scale_factor = self.transform().m11()
            self.initial_scale_factor = self.current_scale_factor
            self.total_translation = QPointF(0, 0)

    def reset_position(self):
        if self.pixmap_item is not None:
            self.setSceneRect(self.pixmap_item.boundingRect())
            self.centerOn(self.pixmap_item)
            self.total_translation = QPointF(0, 0)

    def reset_view(self):
        if self.pixmap_item is not None:
            self.reset_zoom()
            self.reset_position()
