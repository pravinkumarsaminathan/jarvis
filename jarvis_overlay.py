#!/usr/bin/env python3
# File: jarvis_overlay.py
# Description: A macOS-Siri-style always-on-top, translucent overlay for a Jarvis NLP assistant on Linux (Kali).

import sys
import time
import threading
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QFont, QLinearGradient, QColor, QPainterPath

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
    overlay_width: int = 720
    overlay_height: int = 240
    corner_radius: int = 28
    glass_alpha: float = 0.18
    glass_stroke_alpha: float = 0.38
    glow_alpha: float = 0.22
    eq_bar_count: int = 24
    eq_update_ms: int = 28
    hide_on_any_click: bool = True

CFG = Config()

# ==========================
# Signals from background listeners
# ==========================
class Bridge(QtCore.QObject):
    toggle_overlay = Signal()
    start_listening = Signal()
    stop_listening = Signal()
    outside_click = Signal()

# ==========================
# Fancy Glassmorphism Panel
# ==========================
class GlassPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self._radius = CFG.corner_radius

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        path = QPainterPath()
        path.addRoundedRect(10, 10, w - 20, h - 20, self._radius, self._radius)

        glass = QColor(255, 255, 255)
        glass.setAlphaF(CFG.glass_alpha)
        p.fillPath(path, glass)

        grad = QLinearGradient(0, 10, 0, h - 10)
        top = QColor(255, 255, 255, int(255 * 0.12))
        mid = QColor(255, 255, 255, int(255 * 0.05))
        bot = QColor(255, 255, 255, int(255 * 0.10))
        grad.setColorAt(0.0, top)
        grad.setColorAt(0.5, mid)
        grad.setColorAt(1.0, bot)
        p.fillPath(path, grad)

        rim = QColor(130, 200, 255)
        rim.setAlphaF(CFG.glass_stroke_alpha)
        pen = QtGui.QPen(rim, 2.0)
        p.setPen(pen)
        p.drawPath(path)

        glow = QColor(120, 180, 255)
        glow.setAlphaF(CFG.glow_alpha)
        for i in range(1, 6):
            pen = QtGui.QPen(glow, 2.0 + i * 1.8)
            p.setPen(pen)
            p.drawPath(path)

        p.end()

# ==========================
# Animated Equalizer Visualization
# ==========================
class Equalizer(QtWidgets.QWidget):
    feed_levels_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._levels = [0.0] * CFG.eq_bar_count
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(CFG.eq_update_ms)
        self._listening = False
        self._peak_decay = 0.06

        self.feed_levels_signal.connect(self.feed_audio_levels)

    def set_listening(self, on: bool):
        self._listening = on
        if not on:
            self._levels = [0.0] * CFG.eq_bar_count
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
        self._levels = 0.7 * np.array(self._levels) + 0.3 * bands
        self._levels = self._levels.tolist()
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
        margin = 18
        bar_w = max(3, (w - 2 * margin) // (len(self._levels) * 2))
        spacing = bar_w
        x = margin
        base = h - margin
        for lvl in self._levels:
            bar_h = int((h - 2 * margin) * lvl)
            rect = QtCore.QRectF(x, base - bar_h, bar_w, bar_h)
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            c1 = QColor(120, 200, 255, 230)
            c2 = QColor(80, 120, 255, 160)
            grad.setColorAt(0.0, c1)
            grad.setColorAt(1.0, c2)
            p.fillRect(rect, grad)
            x += bar_w + spacing
        p.end()

# ==========================
# Overlay Window
# ==========================
class JarvisOverlay(QtWidgets.QWidget):
    def __init__(self, bridge: Bridge):
        super().__init__()
        self.bridge = bridge
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.Tool
            | Qt.WindowStaysOnTopHint
            | Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating, False)

        self.panel = GlassPanel(self)
        self.title = QtWidgets.QLabel("JARVIS", self)
        self.title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.title.setFont(QFont("SF Pro Display", 22, QFont.Bold))

        self.subtitle = QtWidgets.QLabel("Ready • Press CTRL+S to speak", self)
        self.subtitle.setAlignment(Qt.AlignHCenter)
        self.subtitle.setFont(QFont("Inter", 12))

        self.input = QtWidgets.QLineEdit(self)
        self.input.setPlaceholderText("Type your command… and press Enter")
        self.input.returnPressed.connect(self._submit_text)
        self.input.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.input.setFixedHeight(44)
        self.input.setStyleSheet(
            """
            QLineEdit {
                padding: 0 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.35);
                background: rgba(0,0,0,0.18); color: white; selection-background-color: rgba(130,200,255,0.45);
                font-size: 16px;
            }
            QLineEdit:focus { border: 1px solid rgba(130,200,255,0.9); }
            """
        )

        self.eq = Equalizer(self)
        self.eq.setFixedHeight(68)

        self.mic_hint = QtWidgets.QLabel("Listening…", self)
        self.mic_hint.setAlignment(Qt.AlignHCenter)
        self.mic_hint.setFont(QFont("Inter", 12, QFont.Medium))
        self.mic_hint.setStyleSheet("color: rgba(200,230,255,0.9);")
        self.mic_hint.hide()

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.panel)

        inner = QtWidgets.QVBoxLayout(self.panel)
        inner.setContentsMargins(26, 20, 26, 20)
        inner.setSpacing(10)
        inner.addWidget(self.title)
        inner.addWidget(self.subtitle)
        inner.addWidget(self.input)
        inner.addWidget(self.eq)
        inner.addWidget(self.mic_hint)

        self._listening = False
        self._voice_thread = None
        self._audio_stream = None

        bridge.toggle_overlay.connect(self._toggle)
        bridge.start_listening.connect(self._start_listening)
        bridge.stop_listening.connect(self._stop_listening)
        bridge.outside_click.connect(self._outside_click)

        self.resize(CFG.overlay_width, CFG.overlay_height)
        self._center_top()

    def _center_top(self):
        screen = QtGui.QGuiApplication.primaryScreen().geometry()
        x = int((screen.width() - self.width()) / 2)
        y = int(screen.height() * 0.12)
        self.move(x, y)

    def show_overlay(self):
        self._center_top()
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()

    def hide_overlay(self):
        self._stop_listening()
        self.hide()

    def _toggle(self):
        if self.isVisible():
            self.hide_overlay()
        else:
            self.show_overlay()

    def _start_listening(self):
        if self._listening:
            return
        self._listening = True
        self.eq.set_listening(True)
        self.mic_hint.show()
        self.subtitle.setText("Listening • click to sleep")

        if sd is not None and np is not None:
            self._start_audio_capture()

    def _stop_listening(self):
        if not self._listening:
            return
        self._listening = False
        self.eq.set_listening(False)
        self.mic_hint.hide()
        self.subtitle.setText("Ready • Press CTRL+S to speak")
        self._stop_audio_capture()

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
        self.subtitle.setText(f"You said: {text}")

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._listening or sd is None or np is None:
            return
        samples = np.abs(indata[:, 0]).astype(np.float32)
        samples = np.clip(samples * 4.0, 0.0, 1.0)
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

# ==========================
# Global listeners
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
                if key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
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
                if key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
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

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QColor(18, 20, 28))
        palette.setColor(QtGui.QPalette.WindowText, Qt.white)
        palette.setColor(QtGui.QPalette.Base, QColor(24, 26, 34))
        palette.setColor(QtGui.QPalette.AlternateBase, QColor(28, 30, 38))
        palette.setColor(QtGui.QPalette.Text, Qt.white)
        palette.setColor(QtGui.QPalette.Button, QColor(28, 30, 38))
        palette.setColor(QtGui.QPalette.ButtonText, Qt.white)
        palette.setColor(QtGui.QPalette.Highlight, QColor(120, 180, 255))
        palette.setColor(QtGui.QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        self.bridge = Bridge()
        self.overlay = JarvisOverlay(self.bridge)

        self.tray = QtWidgets.QSystemTrayIcon(QtGui.QIcon())
        self.tray.setToolTip("Jarvis Overlay")
        menu = QtWidgets.QMenu()
        act_toggle = menu.addAction("Toggle (double CTRL)")
        act_toggle.triggered.connect(self.bridge.toggle_overlay.emit)
        act_quit = menu.addAction("Quit")
        act_quit.triggered.connect(self.quit)
        self.tray.setContextMenu(menu)
        self.tray.setVisible(True)

        self.listeners = GlobalListeners(self.bridge)
        self.listeners.start()


def main():
    app = App(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()