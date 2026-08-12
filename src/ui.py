import multiprocessing
import queue
import sys
import os
import json
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QGuiApplication, QMouseEvent, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QScrollArea, QSizePolicy, QFrame, QMainWindow,
    QComboBox, QListView
)
from src.audio import list_audio_devices

# Config file path for persisting preferences
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "preferences.json")


def _load_config():
    """Load saved preferences from disk."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_config(config):
    """Save preferences to disk."""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


class MainWindow(QMainWindow):
    def __init__(self, ui_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, stop_callback, log_path: str):
        super().__init__()
        self.ui_queue = ui_queue
        self.control_queue = control_queue
        self.stop_callback = stop_callback
        self.log_path = log_path
        
        self.transcriber_ready = False
        self.translator_ready = False
        self.is_recording = False
        self._drag_pos = QPoint()
        self._audio_devices = []  # Cached device list

        # Window setup
        self.setWindowTitle("Simultaneous Interpreter")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(650, 600)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Core Font
        font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
        
        # Main Container
        self.container = QWidget()
        self.container.setObjectName("MainContainer")
        self.container.setStyleSheet(f"""
            QWidget#MainContainer {{
                background-color: rgba(15, 23, 42, 220); /* slate-900 with opacity */
                border-radius: 12px;
                border: 1px solid #334155; /* slate-700 */
                font-family: {font_family};
            }}
        """)
        self.setCentralWidget(self.container)
        
        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Area
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        
        # Title & Drag handle
        title_label = QLabel("Simultaneous Interpreter")
        title_label.setStyleSheet("color: #F8FAFC; font-size: 18px; font-weight: 600;")

        # Compact Audio Device Selector (inline in header)
        self.device_combo = QComboBox()
        self.device_combo.setFixedHeight(28)
        self.device_combo.setFixedWidth(200)
        # Use QListView to prevent native macOS dropdown rendering
        popup_view = QListView()
        popup_view.setStyleSheet("""
            QListView {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 4px;
                outline: none;
                font-size: 12px;
            }
            QListView::item {
                min-height: 28px;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QListView::item:hover {
                background-color: #334155;
            }
            QListView::item:selected {
                background-color: #334155;
                color: #F8FAFC;
            }
        """)
        self.device_combo.setView(popup_view)
        self.device_combo.setStyleSheet("""
            QComboBox {
                background-color: #1E293B;
                color: #CBD5E1;
                font-size: 11px;
                padding: 2px 8px;
                border: 1px solid #334155;
                border-radius: 5px;
            }
            QComboBox:hover {
                border-color: #475569;
            }
            QComboBox:disabled {
                color: #475569;
                background-color: #0F172A;
                border-color: #1E293B;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
        """)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        # Compact Refresh Button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedHeight(28)
        self.refresh_btn.setToolTip("Refresh audio devices")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #64748B;
                font-size: 12px;
                padding: 0 10px;
                border: 1px solid transparent;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #F8FAFC;
            }
            QPushButton:disabled {
                color: #1E293B;
            }
        """)
        self.refresh_btn.clicked.connect(self._refresh_devices)

        # Status Label
        self.sys_status_label = QLabel("Loading AI Models...")
        self.sys_status_label.setStyleSheet("color: #94A3B8; font-size: 14px; font-weight: 500;")
        
        # Close Button
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #94A3B8;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #F8FAFC;
            }
        """)
        self.close_btn.clicked.connect(self.close)

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.device_combo)
        header_layout.addWidget(self.refresh_btn)
        header_layout.addSpacing(6)
        header_layout.addWidget(self.sys_status_label)
        header_layout.addSpacing(6)
        header_layout.addWidget(self.close_btn)
        
        main_layout.addLayout(header_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #334155;")
        line.setFixedHeight(1)
        main_layout.addWidget(line)

        # History Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #475569;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        self.history_widget = QWidget()
        self.history_widget.setStyleSheet("background: transparent;")
        self.history_layout = QVBoxLayout()
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_layout.setSpacing(12)
        self.history_widget.setLayout(self.history_layout)
        self.scroll_area.setWidget(self.history_widget)
        main_layout.addWidget(self.scroll_area)

        # Current Message Area
        self.current_label = QLabel("")
        self.current_label.setWordWrap(True)
        self.current_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.current_label.setStyleSheet("""
            color: #CBD5E1; 
            font-size: 18px; 
            padding: 16px; 
            background-color: #1E293B; 
            border: 1px solid #334155;
            border-radius: 8px;
        """)
        # Hide initially since it's empty
        self.current_label.hide()
        main_layout.addWidget(self.current_label)

        # Record Button
        self.action_btn = QPushButton("Loading Models...")
        self.action_btn.setFixedHeight(48)
        self.action_btn.setEnabled(False)
        self.action_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E293B;
                color: #64748B;
                font-size: 16px;
                font-weight: 600;
                border-radius: 8px;
                border: 1px solid #334155;
            }
            QPushButton:enabled {
                background-color: #3B82F6;
                color: #F8FAFC;
                border: none;
            }
            QPushButton:enabled:hover {
                background-color: #2563EB;
            }
            QPushButton[state="recording"] {
                background-color: #EF4444;
                color: #F8FAFC;
                border: none;
            }
            QPushButton[state="recording"]:hover {
                background-color: #DC2626;
            }
        """)
        self.action_btn.clicked.connect(self.on_action_clicked)
        main_layout.addWidget(self.action_btn)
        
        self.setStyleSheet("""
            QFrame#Bubble {
                background-color: transparent;
            }
            QLabel#OrigText {
                color: #94A3B8; /* slate-400 */
                font-size: 14px;
            }
            QLabel#TransText {
                color: #F8FAFC; /* slate-50 */
                font-size: 16px;
                font-weight: 500;
            }
        """)
        
        self._center_on_screen()
        
        # Populate device list and restore saved preference
        self._refresh_devices()
        self._restore_saved_device()
        
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(100)

    # --- Audio Device Management ---

    def _refresh_devices(self):
        """Re-scan available audio devices and populate the combo box."""
        self.device_combo.blockSignals(True)
        current_device_index = None
        if self.device_combo.currentIndex() >= 0 and self._audio_devices:
            current_device_index = self._audio_devices[self.device_combo.currentIndex()]["index"]

        self._audio_devices = list_audio_devices()
        self.device_combo.clear()

        # Type icons for device categorization
        type_icons = {"input": "[IN]", "output": "[OUT]", "both": "[IN/OUT]"}

        selected_combo_index = 0
        for i, dev in enumerate(self._audio_devices):
            icon = type_icons.get(dev.get("type", "input"), "[IN]")
            label = f"{icon} {dev['name']}"
            if dev["is_default"]:
                label = f"{label} [Default]"
            self.device_combo.addItem(label, dev["index"])

            # Restore previous selection if device still exists
            if current_device_index is not None and dev["index"] == current_device_index:
                selected_combo_index = i
            elif current_device_index is None and dev["is_default"]:
                selected_combo_index = i

        if self._audio_devices:
            self.device_combo.setCurrentIndex(selected_combo_index)

        self.device_combo.blockSignals(False)

    def _on_device_changed(self, combo_index):
        """Handle device selection change — sends SET_DEVICE to the audio process."""
        if combo_index < 0 or combo_index >= len(self._audio_devices):
            return
        device = self._audio_devices[combo_index]
        device_index = device["index"]

        # Send SET_DEVICE command to the audio process
        try:
            self.control_queue.put(("SET_DEVICE", device_index), block=False)
        except Exception:
            pass

        # Persist selection
        config = _load_config()
        config["last_device_name"] = device["name"]
        config["last_device_index"] = device_index
        _save_config(config)

    def _restore_saved_device(self):
        """Restore previously selected device from config file."""
        config = _load_config()
        saved_name = config.get("last_device_name")
        if not saved_name:
            return

        # Match by name (device indices can change across reboots)
        for i, dev in enumerate(self._audio_devices):
            if dev["name"] == saved_name:
                if self.device_combo.currentIndex() != i:
                    self.device_combo.setCurrentIndex(i)  # triggers _on_device_changed
                return

    # --- Dragging logic ---

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
            y = geom.y() + (geom.height() - self.height()) // 2
            self.move(x, y)

    def update_system_readiness(self):
        if self.transcriber_ready and self.translator_ready:
            self.sys_status_label.setText("System Ready")
            self.sys_status_label.setStyleSheet("color: #10B981; font-size: 14px; font-weight: 500;") # emerald-500
            if not self.is_recording and self.action_btn.text() == "Loading Models...":
                self.action_btn.setText("Start Recording")
                self.action_btn.setEnabled(True)

    def on_action_clicked(self):
        if not self.is_recording:
            self.is_recording = True
            self.sys_status_label.setText("Recording...")
            self.sys_status_label.setStyleSheet("color: #EF4444; font-size: 14px; font-weight: 500;")
            
            # Disable device switching while recording
            self.device_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
            
            self.action_btn.setProperty("state", "recording")
            self.action_btn.style().unpolish(self.action_btn)
            self.action_btn.style().polish(self.action_btn)
            self.action_btn.setText("Stop Recording")
            
            try:
                self.control_queue.put("START", block=False)
            except Exception:
                pass
        else:
            self.is_recording = False
            self.sys_status_label.setText("Translating...")
            self.sys_status_label.setStyleSheet("color: #F59E0B; font-size: 14px; font-weight: 500;") # amber-500
            
            self.action_btn.setProperty("state", "idle")
            self.action_btn.style().unpolish(self.action_btn)
            self.action_btn.style().polish(self.action_btn)
            self.action_btn.setText("Start Recording")
            self.action_btn.setEnabled(False) # Disable until translation is done
            
            try:
                self.control_queue.put("FINISH", block=False)
            except Exception:
                pass

    def _re_enable_device_controls(self):
        """Re-enable device selector and refresh button after recording/translation completes."""
        self.device_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)

    def _add_to_history(self, original: str, translated: str):
        bubble = QFrame()
        bubble.setObjectName("Bubble")
        bubble_layout = QVBoxLayout()
        bubble_layout.setContentsMargins(12, 12, 12, 12)
        bubble_layout.setSpacing(6)
        bubble.setLayout(bubble_layout)
        
        # A container with a subtle background for the bubble
        bubble.setStyleSheet("""
            QFrame#Bubble {
                background-color: #1E293B;
                border-radius: 8px;
                border: 1px solid #334155;
            }
        """)
        
        lbl_orig = QLabel(f"{original}")
        lbl_orig.setObjectName("OrigText")
        lbl_orig.setWordWrap(True)
        lbl_orig.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        lbl_trans = QLabel(f"{translated}")
        lbl_trans.setObjectName("TransText")
        lbl_trans.setWordWrap(True)
        lbl_trans.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        bubble_layout.addWidget(lbl_orig)
        bubble_layout.addWidget(lbl_trans)
        
        self.history_layout.addWidget(bubble)
        
        # Scroll to bottom smoothly
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def _log_message(self, original: str, translated: str):
        if not self.log_path:
            return
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] Original: {original}\n")
                f.write(f"[{timestamp}] Traducido: {translated}\n")
                f.write("-" * 40 + "\n")
        except Exception:
            pass

    def poll_queue(self):
        try:
            while True:
                item = self.ui_queue.get_nowait()
                if item is None:
                    self.close()
                    return
                
                if isinstance(item, dict):
                    msg_type = item.get("type")
                    
                    if msg_type == "status":
                        if item.get("process") == "transcriber" and item.get("status") == "ready":
                            self.transcriber_ready = True
                        elif item.get("process") == "translator" and item.get("status") == "ready":
                            self.translator_ready = True
                        self.update_system_readiness()
                        
                    elif msg_type == "partial":
                        text = item.get("text", "")
                        if text:
                            self.current_label.show()
                            self.current_label.setText(text)
                            self.current_label.setStyleSheet("""
                                color: #CBD5E1; 
                                font-size: 18px; 
                                padding: 16px; 
                                background-color: #1E293B; 
                                border: 1px solid #334155;
                                border-radius: 8px;
                            """)
                            
                    elif msg_type == "final":
                        text = item.get("text", "")
                        if text:
                            self.current_label.show()
                            self.current_label.setText(f"<i>{text}</i>")
                            self.current_label.setStyleSheet("""
                                color: #94A3B8; 
                                font-style: italic;
                                font-size: 18px; 
                                padding: 16px; 
                                background-color: #1E293B; 
                                border: 1px solid #334155;
                                border-radius: 8px;
                            """)
                            
                    elif msg_type == "translation":
                        original = item.get("original", "")
                        translated = item.get("translated", "")
                        
                        self._add_to_history(original, translated)
                        self._log_message(original, translated)
                        
                        self.current_label.hide()
                        self.current_label.setText("")
                        
                        self.sys_status_label.setText("System Ready")
                        self.sys_status_label.setStyleSheet("color: #10B981; font-size: 14px; font-weight: 500;")
                        
                        self.action_btn.setProperty("state", "idle")
                        self.action_btn.style().unpolish(self.action_btn)
                        self.action_btn.style().polish(self.action_btn)
                        self.action_btn.setText("Start Recording")
                        self.action_btn.setEnabled(True)
                        self._re_enable_device_controls()
                        
                    elif msg_type == "cancel":
                        self.current_label.hide()
                        self.current_label.setText("")
                        
                        self.sys_status_label.setText("System Ready")
                        self.sys_status_label.setStyleSheet("color: #10B981; font-size: 14px; font-weight: 500;")
                        
                        self.action_btn.setProperty("state", "idle")
                        self.action_btn.style().unpolish(self.action_btn)
                        self.action_btn.style().polish(self.action_btn)
                        self.action_btn.setText("Start Recording")
                        self.action_btn.setEnabled(True)
                        self._re_enable_device_controls()

                    elif msg_type == "error":
                        err_msg = item.get("message", "Unknown error")
                        self.current_label.show()
                        self.current_label.setText(f"<b>{err_msg}</b>")
                        self.current_label.setStyleSheet("""
                            color: #F87171; 
                            font-size: 16px; 
                            padding: 16px; 
                            background-color: #450A0A; 
                            border: 1px solid #7F1D1D; 
                            border-radius: 8px;
                        """)
                        
                        self.sys_status_label.setText("System Error")
                        self.sys_status_label.setStyleSheet("color: #EF4444; font-size: 14px; font-weight: 500;")
                        
                        self.action_btn.setProperty("state", "idle")
                        self.action_btn.style().unpolish(self.action_btn)
                        self.action_btn.style().polish(self.action_btn)
                        self.action_btn.setText("Start Recording")
                        self.action_btn.setEnabled(True)
                        self._re_enable_device_controls()
                        self.is_recording = False
                        try:
                            self.control_queue.put("FINISH", block=False)
                        except Exception:
                            pass
                
        except queue.Empty:
            pass

    def closeEvent(self, event):
        self.stop_callback()
        event.accept()

def run_ui(ui_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, start_callback, stop_callback, log_path: str = None):
    app = QApplication(sys.argv)
    
    # Start processes immediately
    start_callback()
    
    window = MainWindow(ui_queue, control_queue, stop_callback, log_path)
    window.show()
    sys.exit(app.exec())
