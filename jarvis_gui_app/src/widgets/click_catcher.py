from PySide6 import QtWidgets, QtCore, QtGui

class ClickCatcher(QtWidgets.QWidget):
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setGeometry(QtGui.QGuiApplication.primaryScreen().geometry())
        self.hide()

    def mousePressEvent(self, event):
        overlay_rect = self.overlay.geometry()
        overlay_pos = self.overlay.mapToGlobal(self.overlay.rect().topLeft())
        overlay_global_rect = QtCore.QRect(overlay_pos, self.overlay.size())
        if not overlay_global_rect.contains(event.globalPosition().toPoint()):
            self.overlay.hide_overlay()
            self.hide()
        else:
            event.ignore()