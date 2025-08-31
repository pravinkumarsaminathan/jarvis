def simple_tokenizer(txt):
    return txt.split()

import os
import sys
from time import time, sleep
from torch.serialization import add_safe_globals
add_safe_globals([simple_tokenizer])

from assistance.core import JarvisAssistant
from assistance.utils import wish_me

from gui.popup_manager import PopupManager

gui_app_path = os.path.join(os.path.dirname(__file__), "jarvis_gui_app", "src")
sys.path.insert(0, gui_app_path)
# Import the OOP GUI application
from jarvis_gui_app.src.app import JarvisApp  # Adjust the import path as per your folder structure

# Path to logo inside data folder
logo_path = os.path.join(os.path.dirname(__file__), "data", "jarvis_logo.png")

def main():
    # Start the OOP GUI
    app = JarvisApp()
    app.run()  # This will start the event loop for your GUI

    # The rest of your logic (if needed)
    # manager = PopupManager()
    # manager.show_popup("Sir I am online, How may i assist you!", logo_path=logo_path, timeout=7000)
    # for i in range(3):
    #     app.processEvents()
    #     app.sendPostedEvents(None, 0)
    #     sleep(1)
    # wish_me()
    # assistant = JarvisAssistant()
    # assistant.run()

if __name__ == "__main__":
    main()