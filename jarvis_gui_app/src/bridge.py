from PySide6 import QtCore

class Bridge(QtCore.QObject):
    toggle_overlay = QtCore.Signal()
    start_listening = QtCore.Signal()
    stop_listening = QtCore.Signal()
    outside_click = QtCore.Signal()

    def __init__(self):
        super().__init__()