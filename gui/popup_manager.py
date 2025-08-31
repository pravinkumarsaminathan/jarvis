import sys
from PySide6 import QtWidgets, QtCore

class PopupManager:
    """Manages popup notifications."""

    def __init__(self, app=None):
        if app is None:
            self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        else:
            self.app = app
        self.active_popups = []

    def show_popup(self, message, logo_path=None, timeout=5000):
        from gui.jarvis_popup import JarvisPopup  # <-- moved import here
        popup = JarvisPopup(message, logo_path, timeout)
        popup.show()
        self.active_popups.append(popup)
        popup.destroyed.connect(lambda: self._remove_popup(popup))

    def _remove_popup(self, popup):
        if popup in self.active_popups:
            self.active_popups.remove(popup)

    def run(self):
        sys.exit(self.app.exec())

    def schedule_popup(self, delay_ms, message, logo_path=None, timeout=5000):
        QtCore.QTimer.singleShot(delay_ms, lambda: self.show_popup(message, logo_path, timeout))