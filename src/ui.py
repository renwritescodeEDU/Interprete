import multiprocessing
import queue
import sys
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout


class OverlayWindow(QWidget):
    def __init__(self, ui_queue: multiprocessing.Queue, log_path: str):
        super().__init__()
        self.ui_queue = ui_queue
        self.log_path = log_path
        self.message_widgets = []
        
        # Window configuration for Overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.setLayout(self.layout)
        
        self.setStyleSheet("""
            QWidget#Bubble {
                background-color: rgba(0, 0, 0, 180);
                border-radius: 12px;
                margin: 2px;
            }
            QLabel#original {
                background-color: transparent;
                color: #A0A0A0;  /* Light gray */
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-size: 18px;
                padding: 5px 10px 0px 10px;
            }
            QLabel#translated {
                background-color: transparent;
                color: #4ADE80;  /* Light green */
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-weight: bold;
                font-size: 24px;
                padding: 0px 10px 10px 10px;
            }
        """)
        
        # Dynamic resizing bounds
        self.setFixedWidth(800)
        self._center_on_screen()
        
        # Polling timer
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(100)

    def _center_on_screen(self):
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + (geom.width() - self.width()) // 2
            # Offset 50 pixels from the bottom
            y = geom.y() + geom.height() - self.height() - 50 
            self.move(x, y)

    def _append_message(self, original: str, translated: str):
        bubble = QWidget()
        bubble.setObjectName("Bubble")
        bubble_layout = QVBoxLayout()
        bubble_layout.setContentsMargins(0, 0, 0, 0)
        bubble.setLayout(bubble_layout)
        
        lbl_orig = QLabel(original)
        lbl_orig.setObjectName("original")
        lbl_orig.setWordWrap(True)
        lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_trans = QLabel(translated)
        lbl_trans.setObjectName("translated")
        lbl_trans.setWordWrap(True)
        lbl_trans.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        bubble_layout.addWidget(lbl_orig)
        bubble_layout.addWidget(lbl_trans)
        
        self.layout.addWidget(bubble)
        self.message_widgets.append(bubble)
        
        # Limit to 4 visual messages
        if len(self.message_widgets) > 4:
            oldest = self.message_widgets.pop(0)
            self.layout.removeWidget(oldest)
            oldest.deleteLater()
            
        self.adjustSize()
        self._center_on_screen()

    def _log_message(self, original: str, translated: str):
        if not self.log_path:
            return
        timestamp = datetime.now().strftime('%H:%M:%S')
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] Original: {original}\n")
            f.write(f"[{timestamp}] Traducido: {translated}\n")
            f.write("-" * 40 + "\n")

    def poll_queue(self):
        try:
            while True:
                item = self.ui_queue.get_nowait()
                if item is None:
                    # Poison pill received from Stop button
                    self.close()
                    return
                
                original, translated = item
                self._append_message(original, translated)
                self._log_message(original, translated)
                
        except queue.Empty:
            pass


class ControlPanelWindow(QWidget):
    def __init__(self, ui_queue: multiprocessing.Queue, start_callback, stop_callback, log_path: str):
        super().__init__()
        self.ui_queue = ui_queue
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.log_path = log_path
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
        self.overlay = OverlayWindow(self.ui_queue, self.log_path)
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


def run_ui(ui_queue: multiprocessing.Queue, start_callback, stop_callback, log_path: str = None):
    app = QApplication(sys.argv)
    window = ControlPanelWindow(ui_queue, start_callback, stop_callback, log_path)
    window.show()
    sys.exit(app.exec())
