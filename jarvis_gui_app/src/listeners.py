import threading
import time
from pynput import keyboard
from PySide6 import QtCore

class GlobalListeners(threading.Thread):
    def __init__(self, bridge, overlay):
        super().__init__(daemon=True)
        self.bridge = bridge
        self.overlay = overlay
        self._last_ctrl_time = 0
        self._ctrl_pressed = False

    def run(self):
        def on_press(key):
            try:
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    now = int(time.time() * 1000)
                    if now - self._last_ctrl_time <= 400:
                        QtCore.QMetaObject.invokeMethod(
                            self.bridge, "toggle_overlay", QtCore.Qt.QueuedConnection
                        )
                        self._last_ctrl_time = 0
                    else:
                        self._last_ctrl_time = now
                    self._ctrl_pressed = True
                elif key == keyboard.KeyCode.from_char('s') and self._ctrl_pressed:
                    # Only allow CTRL+S if overlay is visible and NOT already listening (rectangle mode)
                    if self.overlay.isVisible() and not getattr(self.overlay, "_listening", False):
                        QtCore.QMetaObject.invokeMethod(
                            self.bridge, "start_listening", QtCore.Qt.QueuedConnection
                        )
            except Exception:
                pass

        def on_release(key):
            try:
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    self._ctrl_pressed = False
                if key == keyboard.KeyCode.from_char('s'):
                    if self.overlay.isVisible():
                        QtCore.QMetaObject.invokeMethod(
                            self.bridge, "stop_listening", QtCore.Qt.QueuedConnection
                        )
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        #listener.join() 12