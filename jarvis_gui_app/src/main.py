from PySide6 import QtWidgets
import sys
from app import JarvisApp

def main():
    app = QtWidgets.QApplication(sys.argv)
    jarvis_app = JarvisApp()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()