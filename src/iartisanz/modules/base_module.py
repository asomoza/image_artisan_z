from abc import ABCMeta, abstractmethod

from PyQt6.QtWidgets import QWidget

from iartisanz.app.directories import DirectoriesObject
from iartisanz.app.event_bus import EventBus
from iartisanz.app.preferences import PreferencesObject


class ABCQWidgetMeta(ABCMeta, type(QWidget)):
    pass


class BaseModule(QWidget, metaclass=ABCQWidgetMeta):
    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
    ):
        super().__init__()

        self.directories = directories
        self.preferences = preferences

        self.dialogs = {}

        self.event_bus = EventBus()

    @abstractmethod
    def init_ui(self):
        pass

    def closeEvent(self, event):
        self.event_bus.unsubscribe_all()
        self.event_bus = None
        super().closeEvent(event)
