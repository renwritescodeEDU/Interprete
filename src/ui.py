import multiprocessing
import queue
import sys
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout


class OverlayWindow(QWidget):
    def __init__(self, ui_queue: multiprocessing.Queue):
        super().__init__()
        self.ui_queue = ui_queue
        
        # Window configuration for Overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Layout and styling
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(15, 15, 15, 220);
                border-radius: 12px;
            }
            QLabel {
                background-color: transparent;
                color: #FFFFFF;
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-size: 24px;
                padding: 5px;
            }
            QLabel#translated {
                color: #4ADE80;  /* Light green */
                font-weight: bold;
                font-size: 28px;
            }
        """)
        
        self.original_label = QLabel("")
        self.original_label.setWordWrap(True)
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.translated_label = QLabel("")
        self.translated_label.setObjectName("translated")
        self.translated_label.setWordWrap(True)
        self.translated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.layout.addWidget(self.original_label)
        self.layout.addWidget(self.translated_label)
        
        self.resize(800, 150)
        self._center_on_screen()
        
        # Polling timer
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(100)
        
        # Inactivity timer
        self.clear_timer = QTimer(self)
        self.clear_timer.setSingleShot(True)
        self.clear_timer.timeout.connect(self.clear_text)

    def _center_on_screen(self):
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + (geom.width() - self.width()) // 2
            # Offset 50 pixels from the bottom of the available area (respects the dock)
            y = geom.y() + geom.height() - self.height() - 50 
            self.move(x, y)

    def poll_queue(self):
        try:
            while True:
                item = self.ui_queue.get_nowait()
                if item is None:
                    # Poison pill received from Stop button
                    self.close()
                    return
                
                original, translated = item
                self.original_label.setText(original)
                self.translated_label.setText(translated)
                
                # Restart the clear timer (5 seconds of inactivity)
                self.clear_timer.start(5000)
        except queue.Empty:
            pass
            
    def clear_text(self):
        self.original_label.setText("")
        self.translated_label.setText("")


class ControlPanelWindow(QWidget):
    def __init__(self, ui_queue: multiprocessing.Queue, start_callback, stop_callback):
        super().__init__()
        self.ui_queue = ui_queue
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.overlay = None

        self.setWindowTitle("Interprete - Control Panel")
        self.setFixedSize(300, 150)
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.status_label)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Interpreter")
        self.start_btn.clicked.connect(self.start_interpreter)
        
        self.stop_btn = QPushButton("Stop Interpreter")
        self.stop_btn.clicked.connect(self.stop_interpreter)
        self.stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        self.layout.addLayout(btn_layout)

    def start_interpreter(self):
        self.start_callback()
        self.overlay = OverlayWindow(self.ui_queue)
        self.overlay.show()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Running")

    def stop_interpreter(self):
        self.stop_callback()
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        
        # Clear queue
        while not self.ui_queue.empty():
            try:
                self.ui_queue.get_nowait()
            except:
                break

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Idle")
        
    def closeEvent(self, event):
        self.stop_interpreter()
        event.accept()


def run_ui(ui_queue: multiprocessing.Queue, start_callback, stop_callback):
    app = QApplication(sys.argv)
    window = ControlPanelWindow(ui_queue, start_callback, stop_callback)
    window.show()
    sys.exit(app.exec())
