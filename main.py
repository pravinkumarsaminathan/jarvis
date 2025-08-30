def simple_tokenizer(txt):
    return txt.split()

import os
import sys
from time import time, sleep
from torch.serialization import add_safe_globals
add_safe_globals([simple_tokenizer])

from assistance.core import JarvisAssistant
from assistance.utils import  wish_me

from gui.popup_manager import PopupManager

# Path to logo inside data folder
logo_path = os.path.join(os.path.dirname(__file__), "data", "jarvis_logo.png")


# Path to logo inside data folder
#logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jarvis_logo.png")

def main():
    manager = PopupManager()
    app = manager.app
    manager.show_popup("Sir I am online, How may i assist you!", logo_path=logo_path, timeout=7000)
    app = manager.app
    for i in range(3):
        app.processEvents()   # keeps GUI alive
        app.sendPostedEvents(None, 0)   # ensure button signals get processed
        sleep(1)
    wish_me()
    assistant = JarvisAssistant()
    assistant.run()

if __name__ == "__main__":
    main()