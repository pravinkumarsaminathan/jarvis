from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, Qt
from PySide6.QtCore import Signal
from .glass_panel import GlassPanel
from .perimeter_glow import PerimeterGlow
from .equalizer import Equalizer
from .click_catcher import ClickCatcher
from bridge import Bridge
import sounddevice as sd
import numpy as np
from assistance.core import JarvisAssistant
import threading
import speech_recognition as sr
from PySide6.QtCore import Slot


class JarvisOverlay(QtWidgets.QWidget):
    mic_hint_update = Signal(str)
    input_update = Signal(str)
    stop_listening_signal = Signal()
    subtitle_update = Signal(str)
    reset_subtitle_signal = Signal()

    @Slot(str)
    def _set_mic_hint(self, text):
        self.mic_hint.setText(text)

    @Slot(str)
    def _set_input_text(self, text):
        self.input.setText(text)

    @Slot(str)
    def _set_subtitle(self, text):
        self.subtitle.setText(text)

    @Slot()
    def _reset_subtitle_later(self):
        QtCore.QTimer.singleShot(3000, lambda: self.subtitle.setText("           Press CTRL+S to Speak           "))

    def __init__(self, bridge: Bridge, glow: PerimeterGlow, click_catcher: ClickCatcher):
        super().__init__()
        self.assistant = JarvisAssistant()
        self.glow = glow
        self._audio_stream = None
        self.click_catcher = click_catcher
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(420, 420)
        self._center_on_screen()

        self.installEventFilter(self)

        self.panel = GlassPanel(self)
        self.title = QtWidgets.QLabel("Jarvis", self)
        self.title.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.title.setFont(QtGui.QFont("SF Pro Text", 20, QtGui.QFont.DemiBold))
        self.title.setStyleSheet("color: #fff;")

        self.subtitle = QtWidgets.QLabel("           Press CTRL+S to Speak           ", self)
        self.subtitle.setAlignment(QtCore.Qt.AlignHCenter)
        self.subtitle.setFont(QtGui.QFont("SF Pro Text", 11))
        self.subtitle.setStyleSheet("color: rgba(255,255,255,0.85);")

        self.logo = QtWidgets.QLabel(self)
        self.logo.setAlignment(QtCore.Qt.AlignHCenter)
        self.logo.setPixmap(QtGui.QPixmap("data/jarvis_logo.png").scaled(64, 64, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

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
        self.mic_hint.setAlignment(QtCore.Qt.AlignHCenter)
        self.mic_hint.setFont(QtGui.QFont("SF Pro Text", 12, QtGui.QFont.DemiBold))
        self.mic_hint.setStyleSheet("color: #00ffff;")
        self.mic_hint.hide()

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.panel, alignment=QtCore.Qt.AlignCenter)

        inner = QtWidgets.QVBoxLayout(self.panel)
        inner.setContentsMargins(36, 24, 36, 24)
        inner.setSpacing(10)
        inner.addWidget(self.logo, alignment=QtCore.Qt.AlignHCenter)
        inner.addWidget(self.title, alignment=QtCore.Qt.AlignHCenter)
        inner.addWidget(self.subtitle, alignment=QtCore.Qt.AlignHCenter)
        inner.addWidget(self.eq, alignment=QtCore.Qt.AlignHCenter)
        inner.addWidget(self.input, alignment=QtCore.Qt.AlignHCenter)
        inner.addWidget(self.mic_hint, alignment=QtCore.Qt.AlignHCenter)

        self._listening = False
        self._audio_stream = None

        bridge.toggle_overlay.connect(self._toggle)
        bridge.start_listening.connect(self._start_listening)
        bridge.stop_listening.connect(self._stop_listening)
        bridge.outside_click.connect(self._outside_click)

        self.eq.level_changed.connect(self._on_eq_level)

        self._open_anim = QPropertyAnimation(self, b"pos")
        self._open_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._open_anim.setDuration(380)

        self._close_anim = QPropertyAnimation(self, b"pos")
        self._close_anim.setEasingCurve(QEasingCurve.InCubic)
        self._close_anim.setDuration(300)

        self.hide()

        self._focus_timer = QtCore.QTimer(self)
        self._focus_timer.timeout.connect(self._ensure_input_focus)
        self._focus_timer.setInterval(300)  # ms

        self._voice_thread = None
        self._voice_stop_flag = threading.Event()

        self.mic_hint_update.connect(self.mic_hint.setText)
        self.input_update.connect(self.input.setText)
        self.stop_listening_signal.connect(self._stop_listening)

        self._voice_audio = None
        self._voice_recording = False

        self.subtitle_update.connect(self._set_subtitle)
        self.reset_subtitle_signal.connect(self._reset_subtitle_later)
    
    def _audio_callback(self, indata, frames, time, status):
        # Example: compute audio level and update equalizer
        if status:
            print(status)
        level = float((indata**2).mean())**0.5
        self.eq.level_changed.emit(level)

    def _center_on_screen(self):
        screen = QtGui.QGuiApplication.primaryScreen().geometry()
        x = int((screen.width() - self.width()) / 2)
        y = int((screen.height() - self.height()) / 2)
        self.move(x, y)

    def show_overlay(self):
        self.click_catcher.show()
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()
        self._center_on_screen()
        self._focus_timer.start()

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
        self._focus_timer.stop()

    def _toggle(self):
        if self.isVisible():
            self.hide_overlay()
        else:
            self.show_overlay()

    def _start_listening(self):
        if self._listening:
            return
        self._listening = True
        self.panel.set_listening(True)
        self.resize(420, 420)
        self._center_on_screen()
        self.eq.set_listening(True)
        self.mic_hint.show()
        self.input.hide()
        self.subtitle.setText("Listening — hold CTRL+S and \nspeak")
        self.glow.show_glow()
        if sd is not None and np is not None:
            self._start_audio_capture()
        # Start recording audio
        self._voice_stop_flag.clear()
        self._voice_recording = True
        threading.Thread(target=self._record_voice, daemon=True).start()

    def _stop_listening(self):
        if not self._listening:
            return
        self._listening = False
        self.panel.set_listening(False)
        self.resize(420, 420)
        self._center_on_screen()
        self.eq.set_listening(False)
        self.mic_hint.hide()
        self.input.show()
        self.subtitle_update.emit("Recognizing…")  # Show in rectangle
        self._stop_audio_capture()
        self.glow.set_intensity(0.0)
        self.glow.hide_glow()
        # Start recognition
        threading.Thread(target=self._recognize_voice, daemon=True).start()

    def _outside_click(self):
        if self.isVisible():
            self.hide_overlay()

    def _submit_text(self):
        text = self.input.text().strip()
        if not text:
            return
        self.subtitle.setText("Thinking…")
        self.handle_query(text)
        self.input.clear()

    def handle_query(self, text: str):
        self.subtitle.setText(f"You said: {text.lower().strip()}")
        threading.Thread(target=self.assistant.run, args=(text,), daemon=True).start()
        self.input.show()
        self._ensure_input_focus()

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
        scaled = min(1.0, max(0.02, level * 1.6))
        self.glow.set_intensity(scaled)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            pos = event.position() if hasattr(event, "position") else event.localPos()
            x, y = int(pos.x()), int(pos.y())
            w, h = self.width(), self.height()
            r = 28

            if self._listening:
                cx, cy = w // 2, h // 2
                radius = min(w, h) // 2
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    return True

            else:
                rect = QtCore.QRect(0, 0, w, h)
                if rect.contains(x, y):
                    if (x < r and y < r and (x - r) ** 2 + (y - r) ** 2 > r ** 2) or \
                    (x > w - r and y < r and (x - (w - r)) ** 2 + (y - r) ** 2 > r ** 2) or \
                    (x < r and y > h - r and (x - r) ** 2 + (y - (h - r)) ** 2 > r ** 2) or \
                    (x > w - r and y > h - r and (x - (w - r)) ** 2 + (y - (h - r)) ** 2 > r ** 2):
                        pass
                    else:
                        return True

            if self.isVisible():
                self.hide_overlay()
                return True

        return super().eventFilter(obj, event)
    
    def _ensure_input_focus(self):
        if self.isVisible():
            self.raise_()
            self.activateWindow()
            self.input.setFocus()

    def _voice_input_handler(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            self.mic_hint_update.emit("Listening…")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)
                if self._voice_stop_flag.is_set():
                    self.mic_hint_update.emit("Stopped.")
                    return
                self.mic_hint_update.emit("Recognizing…")
                text = recognizer.recognize_google(audio)
                if not self._voice_stop_flag.is_set():
                    self.stop_listening_signal.emit()
                    self.input_update.emit(text)
                    self.handle_query(text)
            except sr.WaitTimeoutError:
                self.mic_hint_update.emit("Didn't hear anything.")
                self.stop_listening_signal.emit()
            except sr.UnknownValueError:
                self.mic_hint_update.emit("Sorry, I didn't catch that.")
                self.stop_listening_signal.emit()
            except Exception as e:
                self.mic_hint_update.emit(f"Error: {e}")
                self.stop_listening_signal.emit()
    
    def _record_voice(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            self.mic_hint_update.emit("Listening…")
            try:
                # Listen for up to 5 seconds or until silence
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                self._voice_audio = audio
            except Exception as e:
                self._voice_audio = None
                self.mic_hint_update.emit(f"Mic error: {e}")

    def _recognize_voice(self):
        recognizer = sr.Recognizer()
        audio = self._voice_audio
        if audio is None:
            self.mic_hint_update.emit("Didn't hear anything.")
            self.reset_subtitle_signal.emit()
            return
        try:
            text = recognizer.recognize_google(audio)
            cleaned = text.strip().lower()
            if not cleaned:
                self.subtitle_update.emit("try again...")
                self.reset_subtitle_signal.emit()
                return
            self.handle_query(cleaned)
            self.reset_subtitle_signal.emit()
        except sr.UnknownValueError:
            self.mic_hint_update.emit("Sorry, I didn't catch that.")
            self.reset_subtitle_signal.emit()
        except Exception as e:
            self.mic_hint_update.emit(f"Error: {e}")
            self.reset_subtitle_signal.emit()
    
    def _reset_subtitle_later(self, delay_ms=1200):
        QtCore.QTimer.singleShot(delay_ms, lambda: self.subtitle_update.emit("           Press CTRL+S to Speak           "))