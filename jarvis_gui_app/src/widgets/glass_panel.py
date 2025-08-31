from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QLinearGradient, QRadialGradient, QConicalGradient, QPainterPath
from dataclasses import dataclass, field

@dataclass
class GlassConfig:
    border_width: int = 6
    corner_radius: int = 28
    glass_color_top: QColor = field(default_factory=lambda: QColor(20, 20, 25, 240))
    glass_color_bottom: QColor = field(default_factory=lambda: QColor(5, 5, 10, 220))
    inner_glow_color: QColor = field(default_factory=lambda: QColor(100, 200, 255, 35))
    inner_glow_color_mid: QColor = field(default_factory=lambda: QColor(180, 120, 255, 25))

class GlassPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.listening = False
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self._anim_phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(40)
        self.config = GlassConfig()  # <-- Create an instance here

    def set_listening(self, on: bool):
        self.listening = on
        self.update()  # If you want to trigger a repaint or visual change

    def _advance(self):
        self._anim_phase = (self._anim_phase + 0.0025) % 1.0
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        phase = self._anim_phase

        # Compute inset for the path so stroke stays fully inside
        inset = self.config.border_width / 2
        inner_w = w - 2 * inset
        inner_h = h - 2 * inset

        # Box Path
        path = QPainterPath()
        if self.listening:
            size = min(inner_w, inner_h)
            path.addEllipse(inset, inset, size, size)
        else:
            path.addRoundedRect(inset, inset, inner_w, inner_h, self.config.corner_radius, self.config.corner_radius)

        # Glass Core
        glass_grad = QLinearGradient(0, 0, 0, h)
        glass_grad.setColorAt(0.0, self.config.glass_color_top)
        glass_grad.setColorAt(1.0, self.config.glass_color_bottom)
        p.setPen(Qt.NoPen)
        p.setBrush(glass_grad)
        p.drawPath(path)

        # Snake Border
        grad = QConicalGradient(cx, cy, -phase * 360)
        grad.setColorAt(0.00, QColor(0, 255, 255, 255))
        grad.setColorAt(0.05, QColor(120, 80, 255, 220))
        grad.setColorAt(0.10, QColor(255, 0, 200, 180))
        grad.setColorAt(0.18, QColor(255, 200, 80, 140))
        grad.setColorAt(0.35, QColor(0, 0, 0, 0))
        grad.setColorAt(1.00, QColor(0, 0, 0, 0))

        pen = QPen(QBrush(grad), self.config.border_width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

        # Inner Glow
        inner_grad = QRadialGradient(cx, cy, min(inner_w, inner_h) // 1.2)
        inner_grad.setColorAt(0.0, self.config.inner_glow_color)
        inner_grad.setColorAt(0.5, self.config.inner_glow_color_mid)
        inner_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(inner_grad)
        p.setPen(Qt.NoPen)
        p.drawPath(path)