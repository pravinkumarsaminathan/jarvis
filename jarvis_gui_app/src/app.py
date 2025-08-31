from PySide6 import QtWidgets, QtGui
from bridge import Bridge
from widgets.perimeter_glow import PerimeterGlow
from widgets.click_catcher import ClickCatcher
from widgets.jarvis_overlay import JarvisOverlay
from listeners import GlobalListeners

class JarvisApp:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.app.setApplicationName("Jarvis Overlay")
        self.app.setQuitOnLastWindowClosed(False)

        self.bridge = Bridge()
        self.glow = PerimeterGlow()
        self.click_catcher = ClickCatcher(None)
        self.overlay = JarvisOverlay(self.bridge, self.glow, self.click_catcher)
        self.click_catcher.overlay = self.overlay

        # Tray icon
        self.tray = QtWidgets.QSystemTrayIcon(QtGui.QIcon("data/jarvis_logo.png"))
        self.tray.setToolTip("Jarvis Overlay")
        menu = QtWidgets.QMenu()
        act_toggle = menu.addAction("Toggle (double CTRL)")
        act_toggle.triggered.connect(self.bridge.toggle_overlay.emit)
        act_quit = menu.addAction("Quit")
        act_quit.triggered.connect(self.app.quit)
        self.tray.setContextMenu(menu)
        self.tray.setVisible(True)

        # Listeners
        self.listeners = GlobalListeners(self.bridge, self.overlay)
        self.listeners.start()

    def run(self):
        self.app.exec()

    def toggle_overlay(self):
        self.overlay.toggle()  # Toggle the visibility of the overlay

    def start_listening(self):
        self.overlay.start_listening()  # Start the listening process

    def stop_listening(self):
        self.overlay.stop_listening()  # Stop the listening process