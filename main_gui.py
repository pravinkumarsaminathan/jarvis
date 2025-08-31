#!/usr/bin/env python3
# File: jarvis_overlay.py
# Enhanced visuals: glassmorphism, pulsing multi-colored perimeter glow, responsive EQ-linked light, and smooth open/close animations.

import sys
import time
import threading
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QFont, QLinearGradient, QColor, QPainterPath, QRadialGradient, QConicalGradient

from pynput import keyboard, mouse

try:
    import sounddevice as sd
    import numpy as np
except Exception:
    sd = None
    np = None

# ==========================
# Config
# ==========================
@dataclass
class Config:
    double_ctrl_interval_ms: int = 400
    overlay_width: int = 420 #760
    overlay_height: int = 420 #260
    corner_radius: int = 26
    glass_alpha: float = 0.14
    glass_stroke_alpha: float = 0.42
    glow_alpha: float = 0.26
    eq_bar_count: int = 28
    eq_update_ms: int = 28
    hide_on_any_click: bool = True
    glow_cycle_ms: int = 1800

CFG = Config()

# ==========================
# Bridge signals
# ==========================
class Bridge(QtCore.QObject):
    toggle_overlay = QtCore.Signal()
    start_listening = QtCore.Signal()
    stop_listening = QtCore.Signal()
    outside_click = QtCore.Signal()

# ==========================
# Perimeter Glow Widget
# ==========================
class PerimeterGlow(QtWidgets.QWidget):
    """A full-screen, always-on-top translucent widget that paints a multi-colored pulsing glow around screen edges.
    Its intensity/color is controlled via set_intensity(float) where intensity ranges 0.0 - 1.0.
    """

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
        # slowly rotate hue
        self._hue_phase = (self._hue_phase + (40.0 / CFG.glow_cycle_ms)) % 1.0
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

        # Brighter gradient: cyan-magenta-white
        cx, cy = w * 0.5, h * 0.38
        radius = max(w, h) * 0.9
        grad = QRadialGradient(cx, cy, radius)
        alpha = int(255 * self._intensity * CFG.glow_alpha)
        grad.setColorAt(0.00, QColor(0, 255, 255, alpha))   # cyan
        grad.setColorAt(0.35, QColor(255, 0, 255, int(alpha * 0.8)))  # magenta
        grad.setColorAt(0.65, QColor(255, 255, 255, int(alpha * 0.6)))  # white
        grad.setColorAt(1.00, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, grad)
        p.end()

# ==========================
# Glass Panel (main window)
# ==========================
class GlassPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self._anim_phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(40)
        self.listening = False

    def set_listening(self, on: bool):
        self.listening = on
        self.update()

    def _advance(self):
        self._anim_phase = (self._anim_phase + 0.0025) % 1.0  # much slower movement
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        gradient_stops = [
            (0.00, QColor(0, 200, 255, 220)),    # blue
            (0.33, QColor(120, 0, 255, 220)),    # purple
            (0.66, QColor(255, 0, 255, 220)),    # magenta
            (1.00, QColor(0, 200, 255, 220)),    # blue (wrap)
        ]
        phase = self._anim_phase

        import math
        pulse = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        border_base = 2 + int(1 * pulse)
        glow_layers = 8
        glow_strength = 0.18 + 0.10 * pulse

        if self.listening:
            size = min(w, h) - 16
            cx, cy = w // 2, h // 2

            # Drop shadow
            shadow_color = QColor(0, 0, 0, 80)
            for i in range(7, 0, -1):
                p.setPen(Qt.NoPen)
                p.setBrush(shadow_color)
                p.drawEllipse(cx - size // 2 - i, cy - size // 2 - i, size + 2 * i, size + 2 * i)

            # Glassy dark background
            grad = QRadialGradient(cx, cy, size // 2)
            grad.setColorAt(0.0, QColor(40, 40, 60, 220))
            grad.setColorAt(1.0, QColor(10, 10, 20, 200))
            p.setBrush(grad)
            p.setPen(Qt.NoPen)
            p.drawEllipse(cx - size // 2, cy - size // 2, size, size)

            # Subtle frosted overlay
            overlay = QColor(60, 60, 80, 80)
            p.setBrush(overlay)
            p.drawEllipse(cx - size // 2, cy - size // 2, size, size)

            # --- Dynamic Island/Siri Glow Border Effect ---
            for g in range(glow_layers, 0, -1):
                width = border_base + g * 1.5
                alpha_scale = glow_strength * (g / glow_layers)
                border_grad = QConicalGradient(cx, cy, -phase * 360)
                for stop, color in gradient_stops:
                    c = QColor(color)
                    c.setAlphaF(alpha_scale)
                    border_grad.setColorAt(stop, c)
                pen = QtGui.QPen(QtGui.QBrush(border_grad), width)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(
                    cx - size // 2 + width // 2,
                    cy - size // 2 + width // 2,
                    size - width,
                    size - width
                )
            # Final sharp ring, but with a soft gradient
            border_grad = QConicalGradient(cx, cy, -phase * 360)
            for stop, color in gradient_stops:
                border_grad.setColorAt(stop, color)
            pen = QtGui.QPen(QtGui.QBrush(border_grad), border_base)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(
                cx - size // 2 + border_base // 2,
                cy - size // 2 + border_base // 2,
                size - border_base,
                size - border_base
            )
        else:
            r = 32
            shadow_color = QColor(0, 0, 0, 80)
            for i in range(7, 0, -1):
                p.setPen(Qt.NoPen)
                p.setBrush(shadow_color)
                p.drawRoundedRect(8 - i, 8 - i, w - 16 + 2 * i, h - 16 + 2 * i, r + i, r + i)

            grad = QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0.0, QColor(30, 30, 40, 230))
            grad.setColorAt(1.0, QColor(10, 10, 20, 200))
            p.setBrush(grad)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(8, 8, w - 16, h - 16, r, r)

            overlay = QColor(60, 60, 80, 80)
            p.setBrush(overlay)
            p.drawRoundedRect(8, 8, w - 16, h - 16, r, r)

            # --- Ultra-thin border in rectangle mode ---
            border_base_rect = 0.2  # ultra-thin
            for g in range(glow_layers, 0, -1):
                width = border_base_rect + g * 0.4  # even thinner glow
                alpha_scale = glow_strength * (g / glow_layers)
                border_grad = QLinearGradient(8, 8, w - 8, 8)
                for stop, color in gradient_stops:
                    shifted = (stop + phase) % 1.0
                    c = QColor(color)
                    c.setAlphaF(alpha_scale)
                    border_grad.setColorAt(shifted, c)
                for stop, color in gradient_stops:
                    shifted = (stop + phase + 1.0) % 1.0
                    c = QColor(color)
                    c.setAlphaF(alpha_scale)
                    border_grad.setColorAt(shifted, c)
                pen = QtGui.QPen(QtGui.QBrush(border_grad), width)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(8, 8, w - 16, h - 16, r, r)
            border_grad = QLinearGradient(8, 8, w - 8, 8)
            for stop, color in gradient_stops:
                shifted = (stop + phase) % 1.0
                border_grad.setColorAt(shifted, color)
            for stop, color in gradient_stops:
                shifted = (stop + phase + 1.0) % 1.0
                border_grad.setColorAt(shifted, color)
            pen = QtGui.QPen(QtGui.QBrush(border_grad), border_base_rect)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(8, 8, w - 16, h - 16, r, r)

# ==========================
# Equalizer with level signal
# ==========================
class Equalizer(QtWidgets.QWidget):
    feed_levels_signal = QtCore.Signal(object)
    level_changed = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._levels = [0.0] * 12
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(CFG.eq_update_ms)
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
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
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
            grad = QLinearGradient(x, y, x, y - length)
            grad.setColorAt(0.0, QColor(180, 80, 255, 200))   # purple
            grad.setColorAt(0.5, QColor(255, 80, 180, 180))   # pink
            grad.setColorAt(1.0, QColor(80, 220, 255, 160))   # cyan
            p.setPen(Qt.NoPen)
            p.setBrush(grad)
            rect = QtCore.QRectF(x - bar_width // 2, y - length, bar_width, length)
            p.drawRoundedRect(rect, 3, 3)
        p.end()

# ==========================
# Main overlay window with animations
# ==========================
class JarvisOverlay(QtWidgets.QWidget):
    def __init__(self, bridge: Bridge, glow: PerimeterGlow):
        super().__init__()
        self.glow = glow
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(CFG.overlay_width, CFG.overlay_height)
        self._center_on_screen()

        self.panel = GlassPanel(self)
        self.title = QtWidgets.QLabel("Jarvis", self)
        self.title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.title.setFont(QFont("SF Pro Text", 20, QFont.DemiBold))
        self.title.setStyleSheet("color: #fff;")  # white text

        self.subtitle = QtWidgets.QLabel("Ready • Double-press CTRL to open", self)
        self.subtitle.setAlignment(Qt.AlignHCenter)
        self.subtitle.setFont(QFont("SF Pro Text", 11))
        self.subtitle.setStyleSheet("color: rgba(255,255,255,0.85);")  # light text

        self.input = QtWidgets.QLineEdit(self)
        self.input.setPlaceholderText("Type your command… and press Enter")
        self.input.returnPressed.connect(self._submit_text)
        self.input.setFixedHeight(46)
        self.input.setStyleSheet(
            """
            QLineEdit {
                padding: 0 16px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.10);
                background: rgba(30,30,40,0.85); color: #fff; font-size: 15px;
            }
            QLineEdit:focus { border: 1.5px solid #00ffff; }
            """
        )

        self.eq = Equalizer(self)
        self.eq.setFixedHeight(72)

        self.mic_hint = QtWidgets.QLabel("Listening…", self)
        self.mic_hint.setAlignment(Qt.AlignHCenter)
        self.mic_hint.setFont(QFont("SF Pro Text", 12, QFont.DemiBold))
        self.mic_hint.setStyleSheet("color: #00ffff;")  # cyan for visibility
        self.mic_hint.hide()

        # Layout
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.panel, alignment=Qt.AlignCenter)

        inner = QtWidgets.QVBoxLayout(self.panel)
        inner.setContentsMargins(36, 24, 36, 24)
        inner.setSpacing(10)
        inner.addWidget(self.title, alignment=Qt.AlignHCenter)
        inner.addWidget(self.subtitle, alignment=Qt.AlignHCenter)
        inner.addWidget(self.eq, alignment=Qt.AlignHCenter)
        inner.addWidget(self.input, alignment=Qt.AlignHCenter)
        inner.addWidget(self.mic_hint, alignment=Qt.AlignHCenter)

        self._listening = False
        self._audio_stream = None

        # connect bridge
        bridge.toggle_overlay.connect(self._toggle)
        bridge.start_listening.connect(self._start_listening)
        bridge.stop_listening.connect(self._stop_listening)
        bridge.outside_click.connect(self._outside_click)

        # connect eq level to glow
        self.eq.level_changed.connect(self._on_eq_level)

        # Animations
        self._open_anim = QPropertyAnimation(self, b"pos")
        self._open_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._open_anim.setDuration(380)

        self._close_anim = QPropertyAnimation(self, b"pos")
        self._close_anim.setEasingCurve(QEasingCurve.InCubic)
        self._close_anim.setDuration(300)

    def _center_on_screen(self):
        screen = QtGui.QGuiApplication.primaryScreen().geometry()
        x = int((screen.width() - self.width()) / 2)
        y = int((screen.height() - self.height()) / 2)
        self.move(x, y)

    def show_overlay(self):
        # show overlay with slide-down animation (no glow here)
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()
        self._center_on_screen()

        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()

        screen = QtGui.QGuiApplication.primaryScreen().geometry()
        x = int((screen.width() - self.width()) / 2)
        y_center = int((screen.height() - self.height()) / 2)
        start_pos = QtCore.QPoint(x, -self.height() - 40)
        end_pos = QtCore.QPoint(x, y_center)
        self._open_anim.stop()
        self._open_anim.setStartValue(start_pos)
        self._open_anim.setEndValue(end_pos)
        self._open_anim.start()

    def hide_overlay(self):
        # slide-up then hide window (no glow here)
        screen = QtGui.QGuiApplication.primaryScreen().geometry()
        x = int((screen.width() - self.width()) / 2)
        start_pos = self.pos()
        end_pos = QtCore.QPoint(x, -self.height() - 80)
        self._close_anim.stop()
        self._close_anim.setStartValue(start_pos)
        self._close_anim.setEndValue(end_pos)

        def after():
            self._stop_listening()
            self.hide()

        self._close_anim.finished.connect(after)
        self._close_anim.start()

    def _toggle(self):
        if self.isVisible():
            self.hide_overlay()
        else:
            self.show_overlay()

    def _start_listening(self):
        if self._listening:
            return
        self._listening = True
        self.panel.set_listening(True)  # <-- Add this
        self.resize(CFG.overlay_height, CFG.overlay_height)  # Make it a circle
        self._center_on_screen()
        self.eq.set_listening(True)
        self.mic_hint.show()
        self.subtitle.setText("Listening — say something…")
        self.glow.show_glow()
        if sd is not None and np is not None:
            self._start_audio_capture()

    def _stop_listening(self):
        if not self._listening:
            return
        self._listening = False
        self.panel.set_listening(False)  # <-- Add this
        self.resize(CFG.overlay_width, CFG.overlay_height)  # Back to rectangle
        self._center_on_screen()
        self.eq.set_listening(False)
        self.mic_hint.hide()
        self.subtitle.setText("Ready • Double-press CTRL to open")
        self._stop_audio_capture()
        self.glow.set_intensity(0.0)
        self.glow.hide_glow()  # Hide glow when mic stops

    def _outside_click(self):
        if CFG.hide_on_any_click and self.isVisible():
            self.hide_overlay()

    def _submit_text(self):
        text = self.input.text().strip()
        if not text:
            return
        self.subtitle.setText("Thinking…")
        self.handle_query(text)
        self.input.clear()

    def handle_query(self, text: str):
        # Plug your NLP here. For demo we echo.
        self.subtitle.setText(f"You said: {text}")

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._listening or sd is None or np is None:
            return
        samples = np.abs(indata[:, 0]).astype(np.float32)
        samples = np.clip(samples * 4.0, 0.0, 1.0)
        # feed into equalizer via signal (thread-safe)
        self.eq.feed_levels_signal.emit(samples)

    def _start_audio_capture(self):
        if self._audio_stream is not None:
            return
        try:
            self._audio_stream = sd.InputStream(callback=self._audio_callback, channels=1, samplerate=16000)
            self._audio_stream.start()
        except Exception as e:
            self.subtitle.setText(f"Mic error: {e}")

    def _stop_audio_capture(self):
        try:
            if self._audio_stream is not None:
                self._audio_stream.stop()
                self._audio_stream.close()
        finally:
            self._audio_stream = None

    def _on_eq_level(self, level: float):
        # map level (0..1) to glow intensity with slight smoothing
        scaled = min(1.0, max(0.02, level * 1.6))
        self.glow.set_intensity(scaled)

# ==========================
# Global input listeners (pynput)
# ==========================
class GlobalListeners(threading.Thread):
    def __init__(self, bridge: Bridge):
        super().__init__(daemon=True)
        self.bridge = bridge
        self._last_ctrl_time = 0
        self._ctrl_pressed = False
        self._keyboard_listener = None
        self._mouse_listener = None

    def run(self):
        def on_press(key):
            try:
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    now = int(time.time() * 1000)
                    if now - self._last_ctrl_time <= CFG.double_ctrl_interval_ms:
                        self.bridge.toggle_overlay.emit()
                        self._last_ctrl_time = 0
                    else:
                        self._last_ctrl_time = now
                    self._ctrl_pressed = True
                elif key == keyboard.KeyCode.from_char('s') and self._ctrl_pressed:
                    self.bridge.start_listening.emit()
            except Exception:
                pass

        def on_release(key):
            try:
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    self._ctrl_pressed = False
                if key == keyboard.KeyCode.from_char('s'):
                    self.bridge.stop_listening.emit()
            except Exception:
                pass

        def on_click(x, y, button, pressed):
            if pressed:
                self.bridge.outside_click.emit()

        self._keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._mouse_listener = mouse.Listener(on_click=on_click)
        self._keyboard_listener.start()
        self._mouse_listener.start()
        self._keyboard_listener.join()
        self._mouse_listener.join()

# ==========================
# App bootstrap
# ==========================
class App(QtWidgets.QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("Jarvis Overlay")
        self.setDesktopSettingsAware(True)
        self.setStyle("Fusion")
        self.setQuitOnLastWindowClosed(False)

        self.bridge = Bridge()
        self.glow = PerimeterGlow()
        self.overlay = JarvisOverlay(self.bridge, self.glow)

        # tray
        self.tray = QtWidgets.QSystemTrayIcon(QtGui.QIcon("data/jarvis_logo.png"))
        self.tray.setToolTip("Jarvis Overlay")
        menu = QtWidgets.QMenu()
        act_toggle = menu.addAction("Toggle (double CTRL)")
        act_toggle.triggered.connect(self.bridge.toggle_overlay.emit)
        act_quit = menu.addAction("Quit")
        act_quit.triggered.connect(self.quit)
        self.tray.setContextMenu(menu)
        self.tray.setVisible(True)

        # start listeners
        self.listeners = GlobalListeners(self.bridge)
        self.listeners.start()


def main():
    app = App(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()