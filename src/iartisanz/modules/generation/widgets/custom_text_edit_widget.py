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

    def insertTriggerAtCursor(self, text: str, add_comas: bool = False):
        cursor = self.textCursor()
        pos = cursor.position()
        plain = self.toPlainText()
        n = len(plain)

        def char_at(i: int) -> str:
            if 0 <= i < n:
                return plain[i]
            return ""

        prev = char_at(pos - 1)
        prevprev = char_at(pos - 2)
        nextc = char_at(pos)

        prefix = ""
        suffix = ""

        if add_comas:
            if pos > 0:
                if prev == " ":
                    if pos > 1 and prevprev != ",":
                        prefix = ", "
                elif prev != ",":
                    prefix = ", "
                else:
                    prefix = " "

            if pos < n and nextc not in ("", ",") and not nextc.isspace():
                suffix = ", "
        else:
            if pos > 0 and prev not in ("", " ") and not prev.isspace():
                prefix = " "
            if pos < n and nextc not in ("", " ") and not nextc.isspace():
                suffix = " "

        insert_text = f"{prefix}{text}{suffix}"

        cursor.beginEditBlock()
        cursor.setPosition(pos)
        cursor.insertText(insert_text)
        cursor.setPosition(pos + len(insert_text))
        cursor.endEditBlock()
        self.setTextCursor(cursor)
