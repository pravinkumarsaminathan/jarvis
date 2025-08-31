from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QColor, QRadialGradient

class PerimeterGlow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._intensity = 0.0
        self._hue_phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(40)
        self.hide()

    def _advance(self):
        self._hue_phase = (self._hue_phase + (40.0 / 1800)) % 1.0
        self.update()

    def set_intensity(self, v: float):
        v = max(0.0, min(1.0, float(v)))
        if abs(v - self._intensity) > 0.01:
            self._intensity = v
            self.update()

    def show_glow(self):
        self.setGeometry(QtGui.QGuiApplication.primaryScreen().geometry())
        self.show()
        self.raise_()

    def hide_glow(self):
        self.hide()

    def paintEvent(self, event):
        if self._intensity <= 0.001:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        geo = self.geometry()
        w, h = geo.width(), geo.height()

        cx, cy = w * 0.5, h * 0.38
        radius = max(w, h) * 0.9
        grad = QRadialGradient(cx, cy, radius)
        alpha = int(255 * self._intensity * 0.26)
        grad.setColorAt(0.00, QColor(0, 255, 255, alpha))   # cyan
        grad.setColorAt(0.35, QColor(255, 0, 255, int(alpha * 0.8)))  # magenta
        grad.setColorAt(0.65, QColor(255, 255, 255, int(alpha * 0.6)))  # white
        grad.setColorAt(1.00, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, grad)