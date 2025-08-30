# from plyer import notification

# notification.notify(
#     title="J.A.R.V.I.S",
#     message="Sir, I am online. How may I assist you?",
#     timeout=5
# )
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
import sys, os

class JarvisPopup(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setStyleSheet("background-color: #1a1a2e; border: 2px solid #00ffcc; border-radius: 12px;")

        # Layouts
        main_layout = QHBoxLayout()
        text_layout = QVBoxLayout()

        # Avatar / Logo
        logo = QLabel()
        logo_path = "data/jarvis_logo.png"
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setPixmap(pixmap)
        else:
            logo.setText("🤖")  # fallback emoji if logo missing
            logo.setStyleSheet("font-size: 28px; color: #00ffcc;")

        # Message text
        text = QLabel("Sir, I am online.<br>How may I assist you?")
        text.setStyleSheet("color: #00ffcc; font-size: 16px; font-family: Consolas;")

        # Buttons
        btn_layout = QHBoxLayout()

        click_btn = QPushButton("Click me")
        click_btn.setStyleSheet("background-color: #16213e; color: #00ffcc; font-size: 14px; border-radius: 8px; padding: 6px;")
        click_btn.clicked.connect(self.on_click)

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setStyleSheet("background-color: #16213e; color: #ff4c4c; font-size: 14px; border-radius: 8px; padding: 6px;")
        dismiss_btn.clicked.connect(self.close)

        btn_layout.addWidget(click_btn)
        btn_layout.addWidget(dismiss_btn)

        # Arrange layouts
        text_layout.addWidget(text)
        text_layout.addLayout(btn_layout)
        main_layout.addWidget(logo)
        main_layout.addLayout(text_layout)
        self.setLayout(main_layout)

        # Auto close after 10s
        QTimer.singleShot(10000, self.close)

        # Position bottom-right
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 320, screen.height() - 140)
        self.resize(300, 120)

    def on_click(self):
        print("🔔 JARVIS: Button Clicked!")
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    popup = JarvisPopup()
    popup.show()

    # When popup closes, quit app properly
    popup.destroyed.connect(app.quit)

    sys.exit(app.exec_())





