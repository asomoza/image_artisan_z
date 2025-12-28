import logging

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout

from iartisanz.app.directories import DirectoriesObject
from iartisanz.app.preferences import PreferencesObject
from iartisanz.buttons.expand_contract_button import ExpandContractButton
from iartisanz.buttons.vertical_button import VerticalButton
from iartisanz.modules.generation.lora.lora_panel import LoraPanel
from iartisanz.modules.generation.panels.generation_panel import GenerationPanel
from iartisanz.modules.generation.panels.panel_container import PanelContainer


class GenerationRightMenu(QFrame):
    EXPANDED_WIDTH = 400
    NORMAL_WIDTH = 40

    def __init__(self, module_options: dict, preferences: PreferencesObject, directories: DirectoriesObject):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.module_options = module_options
        self.preferences = preferences
        self.directories = directories

        self.expanded = self.module_options.get("right_menu_expanded")
        self.animating = False
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.finished.connect(self.animation_finished)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.setDuration(300)

        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(self.EXPANDED_WIDTH, 50)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        self.panels = {}
        self.current_panel = None
        self.current_panel_text = None

        self.panel_instances: dict[str, QFrame] = {}

        self.init_ui()

        self.add_panel("Generation", GenerationPanel)
        self.add_panel("LoRA", LoraPanel)

        self.loras = []

    def init_ui(self):
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.button_layout = QVBoxLayout()
        self.expand_btn = ExpandContractButton(self.NORMAL_WIDTH, self.NORMAL_WIDTH, True)
        self.expand_btn.clicked.connect(self.on_expand_clicked)
        self.button_layout.addWidget(self.expand_btn)

        self.button_layout.addStretch()
        self.main_layout.addLayout(self.button_layout)

        self.panel_container = PanelContainer()
        self.main_layout.addWidget(self.panel_container)

        self.setLayout(self.main_layout)

    def add_panel(self, text, panel_class, *args):
        button = VerticalButton(text)
        index = self.button_layout.count() - 1
        self.button_layout.insertWidget(index, button)

        self.panels[text] = {
            "class": panel_class,
            "args": (self.module_options, self.preferences, self.directories),
        }

        button.clicked.connect(lambda: self.on_button_clicked(text))

        panel = self.panel_instances.get(text)
        if panel is None:
            panel = panel_class(*self.panels[text]["args"])
            self.panel_container.panel_layout.addWidget(panel)
            self.panel_instances[text] = panel
        panel.setVisible(False)

        if self.current_panel_text is None:
            self.current_panel_text = text
            if self.expanded:
                self.show_panel(text)
            else:
                self.setFixedWidth(self.NORMAL_WIDTH)
                self.expand_btn.extended = False

    def on_button_clicked(self, text):
        self.current_panel_text = text
        self.expand()

    def animation_finished(self):
        self.expanded = not self.expanded

        self.module_options["right_menu_expanded"] = self.expanded
        self.animating = False

    def on_expand_clicked(self):
        if self.expanded:
            self.contract()
        else:
            self.expand()

    def expand(self):
        if self.animating:
            return

        if self.expanded:
            self.show_panel(self.current_panel_text)
        else:
            self.animation.setStartValue(self.NORMAL_WIDTH)
            self.animation.setEndValue(self.EXPANDED_WIDTH)
            self.animation.start()
            self.animating = True

            QTimer.singleShot(
                self.animation.duration(),
                lambda: self.show_panel(self.current_panel_text),
            )

    def contract(self):
        if self.animating:
            return

        if self.current_panel is not None:
            self.current_panel.setVisible(False)

        self.animation.setStartValue(self.EXPANDED_WIDTH)
        self.animation.setEndValue(self.NORMAL_WIDTH)
        self.animation.start()
        self.animating = True

    def show_panel(self, text):
        panel = self.panel_instances.get(text)

        for key, widget in self.panel_instances.items():
            widget.setVisible(key == text)

        self.current_panel = panel
        self.current_panel_text = text

    def closeEvent(self, event):
        for key, panel in self.panel_instances.items():
            panel.setParent(None)
        self.panel_instances.clear()

        super().closeEvent(event)
