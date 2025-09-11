from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QApplication
from PySide6.QtGui import QPixmap, QFont, QColor, QPainter, QBrush
from PySide6.QtCore import Qt, QTimer

class JarvisPopup(QWidget):
    def __init__(self, message, logo_path=None, timeout=5000):
        super().__init__()
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # === Layout ===
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Optional Logo
        if logo_path:
            logo = QLabel()
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                logo.setPixmap(pixmap.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                logo.setAlignment(Qt.AlignCenter)
                layout.addWidget(logo)

        # Message
        label = QLabel(message)
        label.setFont(QFont("Consolas", 13, QFont.Bold))
        label.setStyleSheet("color: white;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)

        self.dismiss_btn = QPushButton("✖ Dismiss")

        self.dismiss_btn.clicked.connect(self.close)

        # Button style
        btn_style = """
            QPushButton {
                background-color: rgba(0, 191, 255, 0.2);
                color: cyan;
                font-weight: bold;
                border: 2px solid cyan;
                border-radius: 10px;
                padding: 6px 18px;
            }
            QPushButton:hover {
                background-color: rgba(0, 191, 255, 0.4);
            }
            QPushButton:pressed {
                background-color: rgba(0, 191, 255, 0.7);
                color: black;
            }
        """
        self.dismiss_btn.setStyleSheet(btn_style)

        btn_row.addWidget(self.dismiss_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

        # Auto-close timer
        if timeout > 0:
            QTimer.singleShot(timeout, self.close)

        # Size & position bottom-right
        self.resize(320, 220)
        screen_geo = QApplication.primaryScreen().availableGeometry()
        x = screen_geo.width() - self.width() - 30
        y = screen_geo.height() - self.height() - 50
        self.move(x, y)

    def paintEvent(self, event):
        """Custom glassmorphic background with neon border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()

        # Semi-transparent glass effect
        brush = QBrush(QColor(30, 30, 30, 180))
        painter.setBrush(brush)
        painter.setPen(QColor(0, 255, 255, 180))  # neon cyan border
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 20, 20)