from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QTextEdit


class CustomTextEditWidget(QTextEdit):
    def __init__(self, parent):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def insertFromMimeData(self, source):
        self.insertPlainText(source.text())

    def insertTextAtCursor(self, text):
        cursor = self.textCursor()
        cursor.insertText(text)

    def insertTriggerAtCursor(self, text):
        cursor = self.textCursor()
        cursor_position = cursor.position()

        if cursor_position > 0:
            cursor.setPosition(cursor_position - 1, QTextCursor.MoveMode.KeepAnchor)
            if cursor.selectedText() == " ":
                if cursor_position > 1:
                    cursor.setPosition(cursor_position - 2, QTextCursor.MoveMode.KeepAnchor)
                    if cursor.selectedText()[0] != ",":
                        text = ", " + text
            elif cursor.selectedText() != ",":
                text = ", " + text
            else:
                text = " " + text

        cursor.setPosition(cursor_position)

        if cursor_position < len(self.toPlainText()):
            cursor.setPosition(cursor_position + 1, QTextCursor.MoveMode.KeepAnchor)
            if cursor.selectedText() != "," and cursor.selectedText() != "":
                text = text + ", "

        cursor.setPosition(cursor_position)
        cursor.insertText(text)
        cursor.setPosition(cursor_position + len(text))
