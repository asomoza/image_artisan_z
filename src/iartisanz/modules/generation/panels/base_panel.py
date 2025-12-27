from PyQt6.QtWidgets import QWidget

from iartisanz.app.directories import DirectoriesObject
from iartisanz.app.event_bus import EventBus
from iartisanz.app.preferences import PreferencesObject


class BasePanel(QWidget):
    def __init__(self, module_options: dict, preferences: PreferencesObject, directories: DirectoriesObject):
        super().__init__()

        self.event_bus = EventBus()

        self.module_options = module_options
        self.preferences = preferences
        self.directories = directories
