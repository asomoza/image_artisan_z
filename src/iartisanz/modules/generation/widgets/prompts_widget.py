import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget
from transformers import Qwen2Tokenizer

from iartisanz.modules.generation.buttons.generate_button import GenerateButton
from iartisanz.modules.generation.widgets.prompt_input import PromptInput


class PromptsWidget(QFrame):
    generate_signal = pyqtSignal(int, str, str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(__name__)

        self.tokenizer = Qwen2Tokenizer.from_pretrained("./configs/Qwen2Tokenizer")
        self.max_tokens = 510

        self.init_ui()
        self.set_button_generate()

    def init_ui(self):
        main_layout = QHBoxLayout()
        text_layout = QVBoxLayout()

        positive_prompt_layout = QGridLayout()
        self.positive_prompt = PromptInput(True, 0, self.max_tokens)
        self.positive_prompt.text_changed.connect(self.on_prompt_changed)
        positive_prompt_layout.addWidget(self.positive_prompt, 0, 0)
        text_layout.addLayout(positive_prompt_layout)

        negative_prompt_layout = QGridLayout()
        self.negative_prompt = PromptInput(False, 0, self.max_tokens)
        self.negative_prompt.text_changed.connect(self.on_prompt_changed)
        negative_prompt_layout.addWidget(self.negative_prompt, 0, 0)
        text_layout.addLayout(negative_prompt_layout)

        main_layout.addLayout(text_layout)

        actions_layout = QVBoxLayout()

        seed_widget = QWidget()
        seed_widget.setMaximumWidth(180)
        seed_layout = QVBoxLayout(seed_widget)
        seed_horizontal_layout = QHBoxLayout()
        seed_label = QLabel("Seed:")
        seed_horizontal_layout.addWidget(seed_label)
        self.seed_text = QLineEdit()
        self.seed_text.setDisabled(True)
        seed_horizontal_layout.addWidget(self.seed_text)
        seed_layout.addLayout(seed_horizontal_layout)
        random_checkbox_layout = QHBoxLayout()
        self.random_checkbox = QCheckBox("Randomize seed")
        self.random_checkbox.setChecked(True)
        self.random_checkbox.clicked.connect(self.randomize_clicked)
        random_checkbox_layout.addWidget(self.random_checkbox)
        random_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        seed_layout.addLayout(random_checkbox_layout)
        actions_layout.addWidget(seed_widget)

        self.generate_button = GenerateButton()
        self.generate_button.setMaximumWidth(180)
        self.generate_button.clicked.connect(self.generate)
        actions_layout.addWidget(self.generate_button)
        actions_layout.setStretch(0, 1)
        actions_layout.setStretch(1, 1)
        main_layout.addLayout(actions_layout)

        main_layout.setStretch(0, 10)
        main_layout.setStretch(1, 2)

        self.setLayout(main_layout)

    def randomize_clicked(self):
        if self.random_checkbox.isChecked():
            self.seed_text.setDisabled(True)
        else:
            self.seed_text.setDisabled(False)

    def on_prompt_changed(self):
        prompt = self.sender()
        text = prompt.toPlainText()

        tokens = self.tokenizer(text).input_ids[1:-1]
        num_tokens = len(tokens)

        prompt.update_token_count(num_tokens)

    def generate(self):
        try:
            seed = int(self.seed_text.text())
        except ValueError:
            seed = -1

        self.generate_signal.emit(seed, self.positive_prompt.toPlainText(), self.negative_prompt.toPlainText())

    def set_button_generate(self):
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #23923b, stop: 1 #124e1f);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2cb349, stop: 1 #176427);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #35d457, stop: 1 #1f8834);
            }
            """
        )
        self.generate_button.setText("Generate")
        self.generate_button.setShortcut("Ctrl+Return")

    def set_button_abort(self):
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #c42b2b, stop: 1 #861e1e);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #db2f2f, stop: 1 #9b2222);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #fa3333, stop: 1 #b62828);
            }
            """
        )
        self.generate_button.setText("Abort")
        self.generate_button.setShortcut("Ctrl+Return")

    def set_button_disable(self):
        self.generate_button.setDisabled(True)
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #8b8989, stop: 1 #5f5f5f);
            }
            """
        )

    def set_button_enable(self):
        self.generate_button.setDisabled(False)
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #23923b, stop: 1 #124e1f);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2cb349, stop: 1 #176427);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #35d457, stop: 1 #1f8834);
            }
            """
        )
