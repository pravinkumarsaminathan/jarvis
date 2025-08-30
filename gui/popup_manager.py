import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from gui.jarvis_popup import JarvisPopup


class PopupManager:
    """Manages popup notifications."""

    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.active_popups = []

    def show_popup(self, message, logo_path=None, timeout=5000):
        popup = JarvisPopup(message, logo_path, timeout)
        popup.show()
        self.active_popups.append(popup)
        popup.destroyed.connect(lambda: self._remove_popup(popup))

    def _remove_popup(self, popup):
        if popup in self.active_popups:
            self.active_popups.remove(popup)

    def run(self):
        sys.exit(self.app.exec_())

    def schedule_popup(self, delay_ms, message, logo_path=None, timeout=5000):
        QTimer.singleShot(delay_ms, lambda: self.show_popup(message, logo_path, timeout))
