"""Messages and questions on top of napari's window, and the guard that asks before napari closes.

Called as ``dialogs.message(...)`` (through the module), so that tests can replace them."""

from __future__ import annotations


def message(text: str, parent=None) -> None:
    """A message the annotator has to close (Done or the next image refused), on top of ``parent``."""
    from qtpy.QtWidgets import QMessageBox

    QMessageBox.information(parent, "AnnoCell", text)


def ask(text: str, parent=None) -> bool:
    """A yes / no question on top of ``parent``; No is the default (Enter does not leave by accident)."""
    from qtpy.QtWidgets import QMessageBox

    buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    answer = QMessageBox.question(parent, "AnnoCell", text, buttons, QMessageBox.StandardButton.No)
    return answer == QMessageBox.StandardButton.Yes


def close_guard(annotator):
    """An event filter for napari's window: closing it with an image not done asks first, and No keeps it open."""
    from qtpy.QtCore import QEvent, QObject

    class CloseGuard(QObject):
        def eventFilter(self, watched, event) -> bool:
            if event.type() == QEvent.Type.Close and not annotator.may_leave("Close napari anyway?"):
                event.ignore()
                return True  # handled: napari does not close
            return False

    return CloseGuard()


def cannot_leave() -> None:
    from napari.utils.notifications import show_error

    show_error("Not saved: this image stays open until it can be saved (is the drive reachable?).")
