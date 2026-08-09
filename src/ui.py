import multiprocessing
import queue
import sys
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QGuiApplication, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QScrollArea, QSizePolicy
)

class OverlayWindow(QWidget):
    def __init__(self, ui_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, log_path: str):
        super().__init__()
        self.ui_queue = ui_queue
        self.control_queue = control_queue
        self.log_path = log_path
        self._drag_pos = QPoint()
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setFixedSize(600, 500)
        
        # Background Container
        self.container = QWidget()
        self.container.setObjectName("MainContainer")
        self.container.setStyleSheet("""
            QWidget#MainContainer {
                background-color: rgba(0, 0, 0, 200);
                border-radius: 12px;
                border: 1px solid #333333;
            }
        """)
        container_layout = QVBoxLayout()
        self.container.setLayout(container_layout)
        self.main_layout.addWidget(self.container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header (Status + Translate Button)
        header_layout = QHBoxLayout()
        self.status_label = QLabel("🔴 Escuchando...")
        self.status_label.setStyleSheet("color: white; font-weight: bold;")
        self.translate_btn = QPushButton("Traducir")
        self.translate_btn.setStyleSheet("background-color: #4ADE80; color: black; font-weight: bold; border-radius: 5px; padding: 5px 15px;")
        self.translate_btn.clicked.connect(self.on_translate_clicked)
        
        header_layout.addWidget(self.status_label)
        header_layout.addStretch()
        header_layout.addWidget(self.translate_btn)
        container_layout.addLayout(header_layout)
        
        # History Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")
        self.history_widget = QWidget()
        self.history_widget.setStyleSheet("background: transparent;")
        self.history_layout = QVBoxLayout()
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_widget.setLayout(self.history_layout)
        self.scroll_area.setWidget(self.history_widget)
        container_layout.addWidget(self.scroll_area)
        
        # Current Message Area (Bottom)
        self.current_label = QLabel("")
        self.current_label.setWordWrap(True)
        self.current_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.current_label.setStyleSheet("color: white; font-size: 24px; padding: 10px; background-color: rgba(255, 255, 255, 10); border-radius: 8px;")
        container_layout.addWidget(self.current_label)
        
        self.setStyleSheet("""
            QWidget#Bubble {
                background-color: rgba(255, 255, 255, 20);
                border-radius: 8px;
                margin-bottom: 5px;
            }
            QLabel.history_text {
                color: #D0D0D0;
                font-size: 16px;
                padding: 5px;
            }
        """)
        
        self._center_on_screen()
        
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(100)

    # Dragging logic
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def _center_on_screen(self):
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + (geom.width() - self.width()) // 2
            y = geom.y() + geom.height() - self.height() - 50 
            self.move(x, y)

    def on_translate_clicked(self):
        self.status_label.setText("⏳ Traduciendo...")
        self.translate_btn.setEnabled(False)
        try:
            self.control_queue.put("FINISH", block=False)
        except:
            pass

    def _add_to_history(self, original: str, translated: str):
        bubble = QWidget()
        bubble.setObjectName("Bubble")
        bubble_layout = QVBoxLayout()
        bubble_layout.setContentsMargins(5, 5, 5, 5)
        bubble.setLayout(bubble_layout)
        
        lbl_orig = QLabel(f"<b>Ori:</b> {original}")
        lbl_orig.setProperty("class", "history_text")
        lbl_orig.setWordWrap(True)
        lbl_orig.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        lbl_trans = QLabel(f"<b>Tr:</b> <span style='color: #4ADE80'>{translated}</span>")
        lbl_trans.setProperty("class", "history_text")
        lbl_trans.setWordWrap(True)
        lbl_trans.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        bubble_layout.addWidget(lbl_orig)
        bubble_layout.addWidget(lbl_trans)
        
        self.history_layout.addWidget(bubble)
        
        # Scroll to bottom
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

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
                    self.close()
                    return
                
                if isinstance(item, dict):
                    msg_type = item.get("type")
                    if msg_type == "partial":
                        self.current_label.setText(item.get("text", ""))
                    elif msg_type == "final":
                        # Clear it while we wait for translation
                        self.current_label.setText(f"<i>{item.get('text', '')}</i>")
                    elif msg_type == "translation":
                        original = item.get("original", "")
                        translated = item.get("translated", "")
                        
                        self._add_to_history(original, translated)
                        self._log_message(original, translated)
                        
                        self.current_label.setText("")
                        self.status_label.setText("🔴 Escuchando...")
                        self.translate_btn.setEnabled(True)
                
        except queue.Empty:
            pass


class ControlPanelWindow(QWidget):
    def __init__(self, ui_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, start_callback, stop_callback, log_path: str):
        super().__init__()
        self.ui_queue = ui_queue
        self.control_queue = control_queue
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
        self.overlay = OverlayWindow(self.ui_queue, self.control_queue, self.log_path)
        self.overlay.show()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Running")

    def stop_interpreter(self):
        self.stop_callback()
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        
        # Clear queues
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


def run_ui(ui_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, start_callback, stop_callback, log_path: str = None):
    app = QApplication(sys.argv)
    window = ControlPanelWindow(ui_queue, control_queue, start_callback, stop_callback, log_path)
    window.show()
    sys.exit(app.exec())
