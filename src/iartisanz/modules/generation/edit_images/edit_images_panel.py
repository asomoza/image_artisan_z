from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from iartisanz.modules.generation.panels.base_panel import BasePanel


class EditImageSlotWidget(QLabel):
    """Single slot showing a 120x120 thumbnail with Edit/Remove buttons."""

    THUMB_SIZE = 120

    def __init__(self, index: int):
        super().__init__()
        self.index = index

        self.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid #555; background: #2a2a2a;")
        self.setText("Empty")

    def set_thumbnail(self, path: str):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.setPixmap(
                pixmap.scaled(
                    self.THUMB_SIZE,
                    self.THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.setText("Empty")

    def clear_thumbnail(self):
        self.clear()
        self.setText("Empty")


class EditImagesPanel(BasePanel):
    MAX_SLOTS = 4

    def __init__(self, *args):
        super().__init__(*args)

        self.thumb_paths: list[str | None] = [None] * self.MAX_SLOTS
        self.thumb_labels: list[EditImageSlotWidget] = []
        self.remove_buttons: list[QPushButton] = []
        self.edit_buttons: list[QPushButton] = []
        self.enable_checkboxes: list[QCheckBox] = []
        self.opacity_effects: list[QGraphicsOpacityEffect] = []

        self._init_ui()

        self.event_bus.subscribe("edit_images", self.on_edit_images_event)

    def _init_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(8)

        title = QLabel("Edit Images")
        title.setStyleSheet("font-weight: 600;")
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(8)

        for i in range(self.MAX_SLOTS):
            slot_layout = QVBoxLayout()
            slot_layout.setSpacing(4)

            thumb = EditImageSlotWidget(i)
            self.thumb_labels.append(thumb)
            slot_layout.addWidget(thumb, alignment=Qt.AlignmentFlag.AlignCenter)

            opacity_effect = QGraphicsOpacityEffect(thumb)
            opacity_effect.setOpacity(1.0)
            thumb.setGraphicsEffect(opacity_effect)
            self.opacity_effects.append(opacity_effect)

            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(4)

            enable_cb = QCheckBox()
            enable_cb.setChecked(True)
            enable_cb.setVisible(False)
            enable_cb.toggled.connect(lambda checked, idx=i: self._on_enable_toggled(idx, checked))
            self.enable_checkboxes.append(enable_cb)
            btn_layout.addWidget(enable_cb)

            edit_btn = QPushButton("Edit")
            edit_btn.setObjectName("green_button")
            edit_btn.setFixedWidth(55)
            edit_btn.clicked.connect(lambda _=False, idx=i: self._on_edit_clicked(idx))
            self.edit_buttons.append(edit_btn)
            btn_layout.addWidget(edit_btn)

            remove_btn = QPushButton("Remove")
            remove_btn.setObjectName("red_button")
            remove_btn.setFixedWidth(55)
            remove_btn.setVisible(False)
            remove_btn.clicked.connect(lambda _=False, idx=i: self._on_remove_clicked(idx))
            self.remove_buttons.append(remove_btn)
            btn_layout.addWidget(remove_btn)

            slot_layout.addLayout(btn_layout)

            row, col = divmod(i, 2)
            grid.addLayout(slot_layout, row, col)

        outer.addLayout(grid)
        outer.addStretch()
        self.setLayout(outer)

    def _on_edit_clicked(self, index: int):
        self.event_bus.publish(
            "manage_dialog",
            {"dialog_type": "edit_images", "action": "open", "image_index": index},
        )

    def _on_remove_clicked(self, index: int):
        self.event_bus.publish(
            "edit_images",
            {"action": "remove", "image_index": index},
        )

    def _on_enable_toggled(self, index: int, checked: bool):
        action = "enable" if checked else "disable"
        self.event_bus.publish(
            "edit_images",
            {"action": action, "image_index": index},
        )

    def _set_slot_dimmed(self, index: int, dimmed: bool):
        self.opacity_effects[index].setOpacity(0.35 if dimmed else 1.0)

    def on_edit_images_event(self, data: dict):
        action = data.get("action")
        index = data.get("image_index")

        if action in ("add", "update"):
            if index is not None and 0 <= index < self.MAX_SLOTS:
                thumb_path = data.get("image_thumb_path")
                self.thumb_paths[index] = thumb_path
                if thumb_path:
                    self.thumb_labels[index].set_thumbnail(thumb_path)
                self.remove_buttons[index].setVisible(True)
                cb = self.enable_checkboxes[index]
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
                cb.setVisible(True)
                self._set_slot_dimmed(index, False)

        elif action == "remove":
            if index is not None and 0 <= index < self.MAX_SLOTS:
                self.thumb_paths[index] = None
                self.thumb_labels[index].clear_thumbnail()
                self.remove_buttons[index].setVisible(False)
                cb = self.enable_checkboxes[index]
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
                cb.setVisible(False)
                self._set_slot_dimmed(index, False)

        elif action == "enable":
            if index is not None and 0 <= index < self.MAX_SLOTS:
                self._set_slot_dimmed(index, False)

        elif action == "disable":
            if index is not None and 0 <= index < self.MAX_SLOTS:
                self._set_slot_dimmed(index, True)

        elif action == "reset":
            for i in range(self.MAX_SLOTS):
                self.thumb_paths[i] = None
                self.thumb_labels[i].clear_thumbnail()
                self.remove_buttons[i].setVisible(False)
                cb = self.enable_checkboxes[i]
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
                cb.setVisible(False)
                self._set_slot_dimmed(i, False)
