#!/usr/bin/env python3
# File: jarvis_overlay.py
# Enhanced visuals: glassmorphism, pulsing multi-colored perimeter glow, responsive EQ-linked light, and smooth open/close animations.

import sys
import time
import threading
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QFont, QLinearGradient, QColor, QPainterPath, QRadialGradient, QConicalGradient, QPen, QBrush

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
        #p.end()

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
        cx, cy = w // 2, h // 2
        phase = self._anim_phase

        # ---------------------------
        # Settings
        # ---------------------------
        border_width = 6
        r = 28

        # Compute inset for the path so stroke stays fully inside
        inset = border_width / 2
        inner_w = w - 2 * inset
        inner_h = h - 2 * inset

        # ---------------------------
        # Box Path (fully inset)
        # ---------------------------
        path = QPainterPath()
        if self.listening:
            size = min(inner_w, inner_h)
            path.addEllipse(inset, inset, size, size)
            grad = QConicalGradient(cx, cy, -phase * 360)
        else:
            path.addRoundedRect(inset, inset, inner_w, inner_h, r, r)
            grad = QConicalGradient(cx, cy, -phase * 360)

        # ---------------------------
        # Glass Core
        # ---------------------------
        glass_grad = QLinearGradient(0, 0, 0, h)
        glass_grad.setColorAt(0.0, QColor(20, 20, 25, 240))
        glass_grad.setColorAt(1.0, QColor(5, 5, 10, 220))
        p.setPen(Qt.NoPen)
        p.setBrush(glass_grad)
        p.drawPath(path)

        # ---------------------------
        # Snake Border (curved head only)
        # ---------------------------
        grad.setColorAt(0.00, QColor(0, 255, 255, 255))
        grad.setColorAt(0.05, QColor(120, 80, 255, 220))
        grad.setColorAt(0.10, QColor(255, 0, 200, 180))
        grad.setColorAt(0.18, QColor(255, 200, 80, 140))
        grad.setColorAt(0.35, QColor(0, 0, 0, 0))
        grad.setColorAt(1.00, QColor(0, 0, 0, 0))

        pen = QPen(QBrush(grad), border_width)
        pen.setCapStyle(Qt.RoundCap)   # curved head
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

        # ---------------------------
        # Inner Glow (optional)
        # ---------------------------
        inner_grad = QRadialGradient(cx, cy, min(inner_w, inner_h) // 1.2)
        inner_grad.setColorAt(0.0, QColor(100, 200, 255, 35))
        inner_grad.setColorAt(0.5, QColor(180, 120, 255, 25))
        inner_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(inner_grad)
        p.setPen(Qt.NoPen)
        p.drawPath(path)





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
    def __init__(self, bridge: Bridge, glow: PerimeterGlow, click_catcher):
        super().__init__()
        self.glow = glow
        self.click_catcher = click_catcher
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(CFG.overlay_width, CFG.overlay_height)
        self._center_on_screen()

        self.installEventFilter(self)

        self.panel = GlassPanel(self)
        self.title = QtWidgets.QLabel("Jarvis", self)
        self.title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.title.setFont(QFont("SF Pro Text", 20, QFont.DemiBold))
        self.title.setStyleSheet("color: #fff;")  # white text

        self.subtitle = QtWidgets.QLabel("Ready • Double-press CTRL to open", self)
        self.subtitle.setAlignment(Qt.AlignHCenter)
        self.subtitle.setFont(QFont("SF Pro Text", 11))
        self.subtitle.setStyleSheet("color: rgba(255,255,255,0.85);")  # light text

        self.logo = QtWidgets.QLabel(self)
        self.logo.setAlignment(Qt.AlignHCenter)
        self.logo.setPixmap(QtGui.QPixmap("data/jarvis_logo.png").scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))


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

        self.mic_hint = QtWidgets.QLabel("…", self)
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
        inner.addWidget(self.logo, alignment=Qt.AlignHCenter)
        inner.addWidget(self.title, alignment=Qt.AlignHCenter)
        #inner.addWidget(self.eq, alignment=Qt.AlignHCenter)
        inner.addWidget(self.subtitle, alignment=Qt.AlignHCenter)
        inner.addWidget(self.eq, alignment=Qt.AlignHCenter)
        inner.addWidget(self.input, alignment=Qt.AlignHCenter)
        inner.addWidget(self.mic_hint, alignment=Qt.AlignHCenter)
        #inner.addWidget(self.eq, alignment=Qt.AlignHCenter)

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
        self.click_catcher.show()
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
            self.click_catcher.hide()

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
        self.input.hide()
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
        self.input.show()
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

    def is_active(self):
        # Returns True if overlay is visible and centered (active)
        return self.isVisible()
    
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            # Get click position in overlay coordinates
            pos = event.position() if hasattr(event, "position") else event.localPos()
            x, y = int(pos.x()), int(pos.y())
            w, h = self.width(), self.height()
            r = 28  # Same as your GlassPanel corner radius

            # If listening, use circle hit test
            if self._listening:
                cx, cy = w // 2, h // 2
                radius = min(w, h) // 2
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    return True  # Click inside circle, do nothing

            # Otherwise, use rounded rectangle hit test
            else:
                rect = QtCore.QRect(0, 0, w, h)
                # Check if inside rounded rect (approximate: inside main rect minus corners)
                if rect.contains(x, y):
                    # Check if in corner
                    if (x < r and y < r and (x - r) ** 2 + (y - r) ** 2 > r ** 2) or \
                    (x > w - r and y < r and (x - (w - r)) ** 2 + (y - r) ** 2 > r ** 2) or \
                    (x < r and y > h - r and (x - r) ** 2 + (y - (h - r)) ** 2 > r ** 2) or \
                    (x > w - r and y > h - r and (x - (w - r)) ** 2 + (y - (h - r)) ** 2 > r ** 2):
                        pass  # In transparent corner, treat as outside
                    else:
                        return True  # Click inside rounded rect, do nothing

            # Otherwise, treat as outside click
            if CFG.hide_on_any_click and self.isVisible():
                self.hide_overlay()
                return True  # Event handled

        return super().eventFilter(obj, event)

# ==========================
# Global input listeners (pynput)
# ==========================
class GlobalListeners(threading.Thread):
    def __init__(self, bridge: Bridge, overlay: JarvisOverlay):
        super().__init__(daemon=True)
        self.bridge = bridge
        self.overlay = overlay
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
                    if self.overlay.is_active():
                        self.bridge.start_listening.emit()
            except Exception:
                pass

        def on_release(key):
            try:
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    self._ctrl_pressed = False
                if key == keyboard.KeyCode.from_char('s'):
                    if self.overlay.is_active():
                        self.bridge.stop_listening.emit()
            except Exception:
                pass

        self._keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._keyboard_listener.start()
        self._keyboard_listener.join()

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
        self.click_catcher = ClickCatcher(None)  # Pass None for now
        self.overlay = JarvisOverlay(self.bridge, self.glow, self.click_catcher)
        self.click_catcher.overlay = self.overlay  # Set overlay reference after creation

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
        self.listeners = GlobalListeners(self.bridge, self.overlay) #Pass overlay
        self.listeners.start()


class ClickCatcher(QtWidgets.QWidget):
    def __init__(self, overlay: QtWidgets.QWidget):
        super().__init__()
        self.overlay = overlay
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(QtGui.QGuiApplication.primaryScreen().geometry())
        self.hide()

    def mousePressEvent(self, event):
        # If click is outside overlay, hide overlay
        overlay_rect = self.overlay.geometry()
        overlay_pos = self.overlay.mapToGlobal(self.overlay.rect().topLeft())
        overlay_global_rect = QtCore.QRect(overlay_pos, self.overlay.size())
        if not overlay_global_rect.contains(event.globalPosition().toPoint()):
            self.overlay.hide_overlay()
            self.hide()
        else:
            event.ignore()

def main():
    app = App(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()