def simple_tokenizer(txt):
    return txt.split()

import os
import sys
import threading
from time import time, sleep
from torch.serialization import add_safe_globals
add_safe_globals([simple_tokenizer])

gui_app_path = os.path.join(os.path.dirname(__file__), "jarvis_gui_app", "src")
sys.path.insert(0, gui_app_path)

# Path to logo inside data folder
logo_path = os.path.join(os.path.dirname(__file__), "data", "jarvis_logo.png")

from PySide6 import QtWidgets

def main():
    # Create ONE QApplication
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Import GUI modules AFTER QApplication is created
    from gui.popup_manager import PopupManager
    from jarvis_gui_app.src.app import JarvisApp
    from assistance.utils import wish_me

    # Pass it to PopupManager
    manager = PopupManager(app=app)
    manager.show_popup("Sir I am online, click double CTRL to access me", logo_path=logo_path, timeout=7000)
    for i in range(2):
        app.processEvents()
        app.sendPostedEvents(None, 0)
        #sleep(1)

    wish_me()
    jarvis_app = JarvisApp()
    jarvis_app.run()

if __name__ == "__main__":
    main()