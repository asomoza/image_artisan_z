from abc import ABCMeta, abstractmethod

from PyQt6.QtWidgets import QStatusBar, QWidget

from iartisanz.app.directories import DirectoriesObject
from iartisanz.app.event_bus import EventBus
from iartisanz.app.preferences import PreferencesObject


class ABCQWidgetMeta(ABCMeta, type(QWidget)):
    pass


class BaseModule(QWidget, metaclass=ABCQWidgetMeta):
    def __init__(
        self,
        status_bar: QStatusBar,
        show_snackbar,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
    ):
        if not isinstance(status_bar, QStatusBar):
            raise TypeError(f"status_bar must be an instance of QStatusBar, not {type(status_bar)}")

        super().__init__()

        self.status_bar = status_bar
        self.show_snackbar = show_snackbar
        self.directories = directories
        self.preferences = preferences

        self.dialogs = {}

        self.event_bus = EventBus()

    @abstractmethod
    def init_ui(self):
        pass

    def update_status_bar(self, text):
        self.status_bar.showMessage(text)

    def closeEvent(self, event):
        self.event_bus.unsubscribe_all()
        self.event_bus = None
        super().closeEvent(event)
