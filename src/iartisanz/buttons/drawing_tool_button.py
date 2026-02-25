from importlib.resources import files

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from .toggle_button import ToggleButton


class DrawingToolButton(QWidget):
    BRUSH_IMG = files("iartisanz.theme.icons").joinpath("brush.png")
    ERASER_IMG = files("iartisanz.theme.icons").joinpath("eraser.png")
    SQUARE_OUTLINE_IMG = files("iartisanz.theme.icons").joinpath("square_outline.png")
    SQUARE_FILL_IMG = files("iartisanz.theme.icons").joinpath("square_fill.png")
    CIRCLE_OUTLINE_IMG = files("iartisanz.theme.icons").joinpath("circle_outline.png")
    CIRCLE_FILL_IMG = files("iartisanz.theme.icons").joinpath("circle_fill.png")
    _clone_stamp_icon = files("iartisanz.theme.icons").joinpath("clone_stamp.png")
    CLONE_STAMP_IMG = _clone_stamp_icon if _clone_stamp_icon.is_file() else BRUSH_IMG

    tool_selected = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()

        self.draw_tool = "brush"
        self.erase_mode = False

        self.init_ui()
        self.on_brush_clicked()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.brush_button = ToggleButton(self.BRUSH_IMG, 25, 25)
        self.brush_button.setToolTip("Brush")
        self.brush_button.clicked.connect(self.on_brush_clicked)
        main_layout.addWidget(self.brush_button)

        self.eraser_button = ToggleButton(self.ERASER_IMG, 25, 25)
        self.eraser_button.setToolTip("Eraser")
        self.eraser_button.clicked.connect(self.on_eraser_clicked)
        main_layout.addWidget(self.eraser_button)

        self.square_outline_button = ToggleButton(self.SQUARE_OUTLINE_IMG, 25, 25)
        self.square_outline_button.setToolTip("Square outline")
        self.square_outline_button.clicked.connect(self.on_square_outline_clicked)
        main_layout.addWidget(self.square_outline_button)

        self.square_fill_button = ToggleButton(self.SQUARE_FILL_IMG, 25, 25)
        self.square_fill_button.setToolTip("Filled square")
        self.square_fill_button.clicked.connect(self.on_square_fill_clicked)
        main_layout.addWidget(self.square_fill_button)

        self.circle_outline_button = ToggleButton(self.CIRCLE_OUTLINE_IMG, 25, 25)
        self.circle_outline_button.setToolTip("Circle outline")
        self.circle_outline_button.clicked.connect(self.on_circle_outline_clicked)
        main_layout.addWidget(self.circle_outline_button)

        self.circle_fill_button = ToggleButton(self.CIRCLE_FILL_IMG, 25, 25)
        self.circle_fill_button.setToolTip("Filled circle")
        self.circle_fill_button.clicked.connect(self.on_circle_fill_clicked)
        main_layout.addWidget(self.circle_fill_button)

        self.clone_stamp_button = ToggleButton(self.CLONE_STAMP_IMG, 25, 25)
        self.clone_stamp_button.setToolTip("Clone stamp (Alt+Click to set source)")
        self.clone_stamp_button.clicked.connect(self.on_clone_stamp_clicked)
        main_layout.addWidget(self.clone_stamp_button)

        self.setLayout(main_layout)

    def _set_button_states(
        self,
        brush: bool,
        eraser: bool,
        square_outline: bool,
        square_fill: bool,
        circle_outline: bool,
        circle_fill: bool,
        clone_stamp: bool,
    ):
        self.brush_button.set_toggle(brush)
        self.eraser_button.set_toggle(eraser)
        self.square_outline_button.set_toggle(square_outline)
        self.square_fill_button.set_toggle(square_fill)
        self.circle_outline_button.set_toggle(circle_outline)
        self.circle_fill_button.set_toggle(circle_fill)
        self.clone_stamp_button.set_toggle(clone_stamp)

    def _emit_tool(self):
        self.tool_selected.emit(self.draw_tool, self.erase_mode)

    def on_brush_clicked(self):
        self._set_button_states(True, False, False, False, False, False, False)
        self.draw_tool = "brush"
        self.erase_mode = False
        self._emit_tool()

    def on_eraser_clicked(self):
        self._set_button_states(False, True, False, False, False, False, False)
        self.draw_tool = "brush"
        self.erase_mode = True
        self._emit_tool()

    def on_square_outline_clicked(self):
        self._set_button_states(False, False, True, False, False, False, False)
        self.draw_tool = "square_outline"
        self.erase_mode = False
        self._emit_tool()

    def on_square_fill_clicked(self):
        self._set_button_states(False, False, False, True, False, False, False)
        self.draw_tool = "square_fill"
        self.erase_mode = False
        self._emit_tool()

    def on_circle_outline_clicked(self):
        self._set_button_states(False, False, False, False, True, False, False)
        self.draw_tool = "circle_outline"
        self.erase_mode = False
        self._emit_tool()

    def on_circle_fill_clicked(self):
        self._set_button_states(False, False, False, False, False, True, False)
        self.draw_tool = "circle_fill"
        self.erase_mode = False
        self._emit_tool()

    def on_clone_stamp_clicked(self):
        self._set_button_states(False, False, False, False, False, False, True)
        self.draw_tool = "clone_stamp"
        self.erase_mode = False
        self._emit_tool()
