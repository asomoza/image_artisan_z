from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

from iartisanz.app.event_bus import EventBus


if TYPE_CHECKING:
    from iartisanz.app.directories import DirectoriesObject
    from iartisanz.app.preferences import PreferencesObject
    from iartisanz.modules.generation.generation_settings import GenerationSettings


class BasePanel(QWidget):
    def __init__(
        self, gen_settings: GenerationSettings, preferences: PreferencesObject, directories: DirectoriesObject
    ):
        super().__init__()

        self.event_bus = EventBus()

        self.gen_settings = gen_settings
        self.preferences = preferences
        self.directories = directories
