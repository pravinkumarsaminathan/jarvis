from PySide6 import QtWidgets, QtCore, QtGui
import numpy as np

class Equalizer(QtWidgets.QWidget):
    feed_levels_signal = QtCore.Signal(object)
    level_changed = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._levels = [0.0] * 12
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(28)  # Update interval in milliseconds
        self._listening = False
        self._peak_decay = 0.06
        self.feed_levels_signal.connect(self.feed_audio_levels)

    def set_listening(self, on: bool):
        self._listening = on
        if not on:
            self._levels = [0.0] * len(self._levels)
            self.update()

    def feed_audio_levels(self, samples):
        if samples is None:
            return
        n = len(self._levels)
        if samples.size < n:
            pad = np.zeros(n - samples.size)
            samples = np.concatenate([samples, pad])
        bands = np.abs(samples[: n]).reshape(n)
        bands = np.clip(bands, 0, 1)
        self._levels = (0.72 * np.array(self._levels) + 0.28 * bands).tolist()
        avg = float(np.mean(self._levels))
        self.level_changed.emit(avg)
        self.update()

    def _tick(self):
        if not self._listening:
            self._levels = [max(0.0, v - self._peak_decay) for v in self._levels]
            self.update()
        else:
            self._levels = [min(1.0, v * 0.98 + 0.01) for v in self._levels]
            self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2 + 10
        radius = min(w, h) // 2 - 10
        bar_count = len(self._levels)
        arc_span = 180  # degrees
        arc_start = 180  # degrees (bottom)
        bar_width = 6  # thinner
        for i, lvl in enumerate(self._levels):
            angle = arc_start + i * (arc_span / (bar_count - 1))
            rad = angle * 3.14159 / 180
            x = cx + radius * np.cos(rad)
            y = cy + radius * np.sin(rad)
            length = 16 + 18 * lvl  # shorter
            grad = QtGui.QLinearGradient(x, y, x, y - length)
            grad.setColorAt(0.0, QtGui.QColor(180, 80, 255, 200))   # purple
            grad.setColorAt(0.5, QtGui.QColor(255, 80, 180, 180))   # pink
            grad.setColorAt(1.0, QtGui.QColor(80, 220, 255, 160))   # cyan
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(grad)
            rect = QtCore.QRectF(x - bar_width // 2, y - length, bar_width, length)
            p.drawRoundedRect(rect, 3, 3)