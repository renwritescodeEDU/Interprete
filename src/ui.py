import multiprocessing
import queue
import sys
import time
import logging
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QGuiApplication, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QScrollArea, QSizePolicy, QFrame, QMainWindow,
    QComboBox, QListView
)
from src.audio import list_audio_devices
from src.config import CONFIG_FILE, load_preferences, save_preferences
from src.events import (
    TYPE_CANCEL,
    TYPE_ERROR,
    TYPE_FINAL,
    TYPE_PARTIAL,
    TYPE_PROVISIONAL,
    TYPE_SKIPPED,
    TYPE_STATUS,
    TYPE_TRANSLATION,
    TYPE_TRUNCATED,
)

# Constants (CONFIG_FILE is imported from src.config)
# Styling Constants
COLOR_BG = "rgba(15, 23, 42, 220)"
COLOR_TEXT_PRIMARY = "#F8FAFC"
COLOR_TEXT_SECONDARY = "#94A3B8"
COLOR_ACCENT = "#3B82F6"
COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"
COLOR_ERROR = "#EF4444"
COLOR_BORDER = "#334155"

# Fallback for truncated-event messages that omit max_minutes.
MAX_RECORDING_MINUTES_DEFAULT = 5
# Maximum number of translation bubbles kept in the history panel.
# Prevents unbounded memory growth over long interpreting sessions.
MAX_HISTORY = 100
# How often the UI re-probes for audio devices while none is available.
# Lets the app self-recover ("waiting for microphone") when a mic is plugged
# in after launch — no restart required.
DEVICE_POLL_INTERVAL_MS = 3000

logger = logging.getLogger(__name__)


def _pipeline_timing_values(timing: dict, now: float):
    """Extract pipeline durations from a timing dict (best-effort).

    Returns (rec_dur, trans_dur, tl_dur, total) or (0, 0, 0, 0) if
    required keys are missing.
    """
    if not timing.get("recording_start") or not timing.get("recording_stop"):
        return 0.0, 0.0, 0.0, 0.0
    rec_dur = timing["recording_stop"] - timing["recording_start"]
    trans_dur = timing.get("transcription_end", 0) - timing.get("transcription_start", 0) if timing.get("transcription_start") else 0
    tl_dur = timing.get("translation_end", 0) - timing.get("translation_start", 0) if timing.get("translation_start") else 0
    total = now - timing["recording_stop"]
    return rec_dur, trans_dur, tl_dur, total

def _load_config():
    """Load saved preferences from disk (delegates to src.config)."""
    return load_preferences(CONFIG_FILE)


def _save_config(config):
    """Save preferences to disk (delegates to src.config)."""
    save_preferences(config, CONFIG_FILE)

class MainWindow(QMainWindow):
    def __init__(self, ui_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, stop_callback, log_path: str, health_check=None):
        super().__init__()
        self.ui_queue = ui_queue
        self.control_queue = control_queue
        self.stop_callback = stop_callback
        self.log_path = log_path
        self.health_check = health_check
        
        self.transcriber_ready = False
        self.translator_ready = False
        self.audio_ready = False
        self._audio_waiting = False
        self.is_recording = False
        self._event_handlers = self._build_event_handlers()
        self._drag_pos = QPoint()
        self._audio_devices = []
        self._workers_dead = False
        self._health_misses = 0
        self._shutting_down = False

        self._setup_ui()
        self._center_on_screen()
        
        self._refresh_devices()
        self._restore_saved_device()
        
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(100)

        # Watchdog: periodically verify worker process liveness.
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.check_health)
        self.health_timer.start(3000)

        # Device polling: while no microphone is available, periodically
        # re-enumerate devices so a freshly connected mic is picked up
        # automatically (and the audio worker is told to re-probe).
        self.device_timer = QTimer(self)
        self.device_timer.timeout.connect(self._poll_devices)
        self.device_timer.start(DEVICE_POLL_INTERVAL_MS)

    def _setup_ui(self):
        self.setWindowTitle("Simultaneous Interpreter")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(650, 600)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Qt-native font resolution: "Segoe UI" on Windows, system font on
        # macOS. Uses the resolved application font (not a bare QFont, whose
        # unset point size triggers "QFont::setPointSize: Point size <= 0"
        # warnings).
        font_family = QApplication.font().family()
        self.container = QWidget()
        self.container.setObjectName("MainContainer")
        self.container.setStyleSheet(f"""
            QWidget#MainContainer {{
                background-color: {COLOR_BG};
                border-radius: 12px;
                border: 1px solid {COLOR_BORDER};
                font-family: {font_family};
            }}
        """)
        self.setCentralWidget(self.container)
        
        # Set a concrete application font with a valid size so the stylesheet
        # `font-family` never triggers "QFont::setPointSize: Point size <= 0".
        base_font = QApplication.font()
        base_font.setPointSize(13)
        QApplication.setFont(base_font)
        
        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        self._setup_header(main_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {COLOR_BORDER};")
        line.setFixedHeight(1)
        main_layout.addWidget(line)

        self._setup_history_area(main_layout)
        self._setup_action_area(main_layout)
        
        self.setStyleSheet("""
            QFrame#Bubble { background-color: transparent; }
            QLabel#OrigText { color: #94A3B8; font-size: 14px; }
            QLabel#TransText { color: #F8FAFC; font-size: 16px; font-weight: 500; }
        """)

    def _setup_header(self, main_layout):
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        
        title_label = QLabel("Simultaneous Interpreter")
        title_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 18px; font-weight: 600;")

        self.device_combo = QComboBox()
        self.device_combo.setFixedHeight(28)
        self.device_combo.setFixedWidth(200)
        popup_view = QListView()
        popup_view.setStyleSheet("""
            QListView { background-color: #1E293B; color: #E2E8F0; border: 1px solid #475569; border-radius: 6px; padding: 4px; font-size: 12px; }
            QListView::item { min-height: 28px; padding: 4px 8px; border-radius: 4px; }
            QListView::item:hover { background-color: #334155; }
            QListView::item:selected { background-color: #334155; color: #F8FAFC; }
        """)
        self.device_combo.setView(popup_view)
        self.device_combo.setStyleSheet("""
            QComboBox { background-color: #1E293B; color: #CBD5E1; font-size: 11px; padding: 2px 8px; border: 1px solid #334155; border-radius: 5px; }
            QComboBox:hover { border-color: #475569; }
            QComboBox:disabled { color: #475569; background-color: #0F172A; border-color: #1E293B; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { image: none; border: none; }
        """)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedHeight(28)
        self.refresh_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #64748B; font-size: 12px; padding: 0 10px; border-radius: 4px; }
            QPushButton:hover { background-color: #334155; color: #F8FAFC; }
            QPushButton:disabled { color: #1E293B; }
        """)
        self.refresh_btn.clicked.connect(self._refresh_devices)

        self.sys_status_label = QLabel("Loading AI Models...")
        self.sys_status_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 14px; font-weight: 500;")
        
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #94A3B8; font-size: 16px; font-weight: bold; border: none; border-radius: 4px; }
            QPushButton:hover { background-color: #334155; color: #F8FAFC; }
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

    def _setup_history_area(self, main_layout):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { border: none; background: transparent; width: 8px; margin: 0px; }
            QScrollBar::handle:vertical { background: #475569; min-height: 20px; border-radius: 4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        self.history_widget = QWidget()
        self.history_widget.setStyleSheet("background: transparent;")
        self.history_layout = QVBoxLayout()
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_layout.setSpacing(12)
        self.history_widget.setLayout(self.history_layout)
        self.scroll_area.setWidget(self.history_widget)
        main_layout.addWidget(self.scroll_area)

    def _setup_action_area(self, main_layout):
        self.current_label = QLabel("")
        self.current_label.setWordWrap(True)
        self.current_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.current_label.hide()
        main_layout.addWidget(self.current_label)

        self.action_btn = QPushButton("Loading Models...")
        self.action_btn.setFixedHeight(48)
        self.action_btn.setEnabled(False)
        self.action_btn.setStyleSheet(f"""
            QPushButton {{ background-color: #1E293B; color: #64748B; font-size: 16px; font-weight: 600; border-radius: 8px; border: 1px solid #334155; }}
            QPushButton:enabled {{ background-color: {COLOR_ACCENT}; color: {COLOR_TEXT_PRIMARY}; border: none; }}
            QPushButton:enabled:hover {{ background-color: #2563EB; }}
            QPushButton[state="recording"] {{ background-color: {COLOR_ERROR}; color: {COLOR_TEXT_PRIMARY}; border: none; }}
            QPushButton[state="recording"]:hover {{ background-color: #DC2626; }}
        """)
        self.action_btn.clicked.connect(self.on_action_clicked)
        main_layout.addWidget(self.action_btn)

    def _refresh_devices(self):
        self.device_combo.blockSignals(True)
        current_device_index = None
        if self.device_combo.currentIndex() >= 0 and self._audio_devices:
            current_device_index = self._audio_devices[self.device_combo.currentIndex()]["index"]

        # Only input-capable devices can be recording sources. Filtering here
        # prevents the combo from landing on an output-only device when no
        # microphone is present (a common Windows failure mode).
        self._audio_devices = [
            dev for dev in list_audio_devices()
            if dev.get("type") in ("input", "both")
        ]
        self.device_combo.clear()
        type_icons = {"input": "[IN]", "output": "[OUT]", "both": "[IN/OUT]"}

        selected_combo_index = 0
        for i, dev in enumerate(self._audio_devices):
            icon = type_icons.get(dev.get("type", "input"), "[IN]")
            label = f"{icon} {dev['name']}{' [Default]' if dev['is_default'] else ''}"
            self.device_combo.addItem(label, dev["index"])
            if current_device_index is not None and dev["index"] == current_device_index:
                selected_combo_index = i
            elif current_device_index is None and dev["is_default"]:
                selected_combo_index = i

        if self._audio_devices:
            self.device_combo.setCurrentIndex(selected_combo_index)
            self.device_combo.setEnabled(True)
            self.device_combo.blockSignals(False)
            self._audio_waiting = False
            self.update_system_readiness()
        else:
            # No input device available: keep the combo disabled and surface
            # the recoverable "waiting for microphone" state.
            self.device_combo.setEnabled(False)
            self.device_combo.blockSignals(False)
            self._audio_waiting = True
            self.update_system_readiness()
            self._maybe_show_audio_waiting()

    def _poll_devices(self):
        """Auto-recover when no microphone was present at launch.

        Runs on a timer and is a no-op once an input device exists or while
        recording, so it never interferes with a manual device selection.
        Enumeration is throttled (at most once per DEVICE_POLL_INTERVAL_MS)
        so a machine with many virtual devices (VoiceMeeter/VB-CABLE) is not
        re-scanned on every tick — the audio worker already re-probes in its
        own process.
        """
        if self.is_recording or self._audio_devices:
            return
        now = time.time()
        if getattr(self, "_last_device_poll", 0) and (now - self._last_device_poll) < DEVICE_POLL_INTERVAL_MS / 1000:
            return
        self._last_device_poll = now
        logger.info("[UI] No input device available — re-probing for a microphone...")
        self._refresh_devices()
        if self._audio_devices:
            # A device appeared: point the audio worker at it immediately.
            self._on_device_changed(self.device_combo.currentIndex())

    def _maybe_show_audio_waiting(self):
        """Banner shown while the pipeline waits for an input device."""
        if not self._audio_waiting or self.is_recording:
            return
        self.current_label.show()
        self.current_label.setText(
            "<b>Waiting for microphone...</b> Connect a microphone (or a virtual "
            "audio device) — it will be detected automatically."
        )
        self.current_label.setStyleSheet(
            f"color: {COLOR_WARNING}; font-size: 15px; padding: 16px; "
            f"background-color: #451A03; border: 1px solid #92400E; border-radius: 8px;"
        )

    def _on_device_changed(self, combo_index):
        if combo_index < 0 or combo_index >= len(self._audio_devices):
            return
        device = self._audio_devices[combo_index]
        try:
            self.control_queue.put(("SET_DEVICE", device["index"]), block=False)
        except Exception as e:
            logger.warning(f"Failed to queue SET_DEVICE: {e}")

        config = _load_config()
        config["last_device_name"] = device["name"]
        config["last_device_index"] = device["index"]
        _save_config(config)

    def _restore_saved_device(self):
        config = _load_config()
        saved_name = config.get("last_device_name")
        if not saved_name:
            return
        for i, dev in enumerate(self._audio_devices):
            if dev["name"] == saved_name:
                if self.device_combo.currentIndex() != i:
                    self.device_combo.setCurrentIndex(i)
                return

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
        else:
            # Fallback for headless / RDP environments where the screen
            # geometry is unavailable.
            self.move(0, 0)

    def update_system_readiness(self):
        models_ready = self.transcriber_ready and self.translator_ready
        if models_ready and not self.audio_ready:
            # No stream confirmed by the audio worker yet. Distinguish the
            # "no device at all" case from the brief "connecting" window after
            # a device is detected.
            if self._audio_devices:
                self.sys_status_label.setText("Connecting to microphone...")
            else:
                self.sys_status_label.setText("Waiting for microphone...")
            self.sys_status_label.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 14px; font-weight: 500;")
            if not self.is_recording:
                self.action_btn.setEnabled(False)
            return
        if models_ready and self.audio_ready:
            self.sys_status_label.setText("System Ready")
            self.sys_status_label.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 14px; font-weight: 500;")
            if not self.is_recording and self.action_btn.text() == "Loading Models...":
                self.action_btn.setText("Start Recording")
                self.action_btn.setEnabled(True)

    def check_health(self):
        """Watchdog: flag a worker that died without sending an error event."""
        if self._shutting_down:
            return  # processes are being stopped intentionally — never a crash
        if not self.health_check:
            return
        if not (self.transcriber_ready or self.translator_ready):
            return  # workers not fully started yet
        status = self.health_check()
        dead = [name for name, alive in status.items() if not alive]
        if dead:
            # Require 2 consecutive misses to avoid false positives during
            # transient states (e.g. a process briefly between restarts).
            self._health_misses += 1
            if self._health_misses >= 2 and not self._workers_dead:
                self._workers_dead = True
                logger.error(f"[UI] Worker crash detected: {', '.join(dead)}")
                self.current_label.show()
                self.current_label.setText(f"<b>Worker crash:</b> {', '.join(dead)}. Restart the app.")
                self.current_label.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 16px; padding: 16px; background-color: #450A0A; border: 1px solid #7F1D1D; border-radius: 8px;")
                self.sys_status_label.setText("System Error")
                self.sys_status_label.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 14px; font-weight: 500;")
                self.action_btn.setEnabled(False)
        else:
            self._health_misses = 0
            self._workers_dead = False

    def on_action_clicked(self):
        if not self.is_recording:
            self.is_recording = True
            self._recording_start_time = time.time()
            logger.info("[UI] Start Recording clicked")
            self.sys_status_label.setText("Recording...")
            self.sys_status_label.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 14px; font-weight: 500;")
            self.device_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
            self.action_btn.setProperty("state", "recording")
            self.action_btn.style().unpolish(self.action_btn)
            self.action_btn.style().polish(self.action_btn)
            self.action_btn.setText("Stop Recording")
            try:
                self.control_queue.put(("START", self._recording_start_time), block=False)
            except Exception as e:
                logger.warning(f"Failed to queue START: {e}")
        else:
            stop_time = time.time()
            recording_duration = stop_time - self._recording_start_time if hasattr(self, '_recording_start_time') else 0
            logger.info(f"[UI] Stop Recording clicked (recorded for {recording_duration:.2f}s)")
            self.is_recording = False
            self.sys_status_label.setText("Translating...")
            self.sys_status_label.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 14px; font-weight: 500;")
            self.action_btn.setProperty("state", "idle")
            self.action_btn.style().unpolish(self.action_btn)
            self.action_btn.style().polish(self.action_btn)
            self.action_btn.setText("Start Recording")
            self.action_btn.setEnabled(False)
            try:
                self.control_queue.put(("FINISH", stop_time), block=False)
            except Exception as e:
                logger.warning(f"Failed to queue FINISH: {e}")

    def _reset_ui_state(self, is_error=False):
        """Deduplicates common UI reset code."""
        if not is_error:
            self.current_label.hide()
            self.current_label.setText("")
            self.update_system_readiness()
        self.action_btn.setProperty("state", "idle")
        self.action_btn.style().unpolish(self.action_btn)
        self.action_btn.style().polish(self.action_btn)
        self.action_btn.setText("Start Recording")
        self.action_btn.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        if self._audio_waiting and not is_error:
            # Re-apply the waiting banner and keep the record button disabled;
            # otherwise update_system_readiness() re-enables it above. On error
            # paths the error banner in current_label must stay visible.
            self.action_btn.setEnabled(False)
            self.device_combo.setEnabled(False)
            self._maybe_show_audio_waiting()

    def _add_to_history(self, original: str, translated: str, latency: float = 0.0):
        # Cap history to bound memory growth on long sessions
        while self.history_layout.count() >= MAX_HISTORY:
            oldest = self.history_layout.takeAt(0)
            if oldest and oldest.widget():
                oldest.widget().deleteLater()

        bubble = QFrame()
        bubble.setObjectName("Bubble")
        bubble_layout = QVBoxLayout()
        bubble_layout.setContentsMargins(12, 12, 12, 12)
        bubble_layout.setSpacing(6)
        bubble.setLayout(bubble_layout)
        bubble.setStyleSheet("QFrame#Bubble { background-color: #1E293B; border-radius: 8px; border: 1px solid #334155; }")
        
        lbl_orig = QLabel(original)
        lbl_orig.setObjectName("OrigText")
        lbl_orig.setWordWrap(True)
        lbl_orig.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        lbl_trans = QLabel(translated)
        lbl_trans.setObjectName("TransText")
        lbl_trans.setWordWrap(True)
        lbl_trans.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        bubble_layout.addWidget(lbl_orig)
        bubble_layout.addWidget(lbl_trans)
        self.history_layout.addWidget(bubble)
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum()))

    def _log_message(self, original: str, translated: str, timing: dict = None):
        if not self.log_path:
            return
        try:
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            lines = [
                f"[{timestamp}] Original: {original}",
                f"[{timestamp}] Traducido: {translated}",
            ]
            if timing and timing.get("recording_start") and timing.get("recording_stop"):
                rec_dur, trans_dur, tl_dur, total = _pipeline_timing_values(timing, time.time())
                status_tag = "OK" if total <= 2.0 else "SLOW"
                lines.append(f"[{timestamp}] Pipeline: rec={rec_dur:.2f}s | asr={trans_dur:.3f}s | tl={tl_dur:.3f}s | total={total:.3f}s [{status_tag}]")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n" + "-" * 40 + "\n")
        except Exception as e:
            logger.warning(f"Failed to log message: {e}")

    def poll_queue(self):
        try:
            while True:
                item = self.ui_queue.get_nowait()
                if item is None:
                    self.close()
                    return
                if isinstance(item, dict):
                    handler = self._event_handlers.get(item.get("type"))
                    if handler is not None:
                        handler(item)
        except queue.Empty:
            pass

    def _build_event_handlers(self) -> dict:
        """Maps event types to their UI handlers (single dispatch point)."""
        return {
            TYPE_STATUS: self._handle_status,
            TYPE_PARTIAL: self._handle_partial,
            TYPE_FINAL: self._handle_final,
            TYPE_PROVISIONAL: self._handle_provisional,
            TYPE_TRANSLATION: self._handle_translation,
            TYPE_CANCEL: self._handle_cancel,
            TYPE_SKIPPED: self._handle_skipped,
            TYPE_TRUNCATED: self._handle_truncated,
            TYPE_ERROR: self._handle_error,
        }

    def _handle_status(self, item):
        process = item.get("process")
        status = item.get("status")
        if process == "audio" and status == "waiting":
            self.audio_ready = False
            self._audio_waiting = True
            if self.is_recording:
                # Device dropped mid-recording: the worker can no
                # longer capture, so release the UI recording state
                # instead of leaving a dead "Stop Recording" button.
                self.is_recording = False
                self.action_btn.setProperty("state", "idle")
                self.action_btn.style().unpolish(self.action_btn)
                self.action_btn.style().polish(self.action_btn)
                self.action_btn.setText("Start Recording")
            # Invalidate the cached device list so the poll timer
            # re-enumerates (a different mic may be connected now).
            self._audio_devices = []
            self.device_combo.clear()
            self.device_combo.setEnabled(False)
            self.update_system_readiness()
            self._maybe_show_audio_waiting()
        elif process == "audio" and status == "ready":
            self.audio_ready = True
            was_waiting = self._audio_waiting
            self._audio_waiting = False
            if was_waiting:
                # Clear the "Waiting for microphone..." banner now
                # that a stream is confirmed.
                self.current_label.hide()
                self.current_label.setText("")
            if not self._audio_devices:
                # The worker opened a stream, so devices exist;
                # populate the combo if our earlier probe saw none.
                self._refresh_devices()
            self.update_system_readiness()
        elif process == "transcriber" and status == "ready":
            self.transcriber_ready = True
            self.update_system_readiness()
        elif process == "translator" and status == "ready":
            self.translator_ready = True
            self.update_system_readiness()
        elif process == "translator" and status == "ollama_waiting":
            self.sys_status_label.setText("Starting Ollama...")
            self.sys_status_label.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 14px; font-weight: 500;")
        elif process == "translator" and status == "ollama_offline":
            self.sys_status_label.setText("Ollama offline")
            self.sys_status_label.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 14px; font-weight: 500;")
            self.current_label.show()
            self.current_label.setText(
                "<b>Ollama is not running.</b> Start Ollama (or install it from "
                "https://ollama.com/download) — the app will recover automatically."
            )
            self.current_label.setStyleSheet(
                f"color: {COLOR_ERROR}; font-size: 15px; padding: 16px; "
                f"background-color: #450A0A; border: 1px solid #7F1D1D; border-radius: 8px;"
            )
        elif process == "translator" and status == "model_download":
            model = item.get("model", "the translation model")
            self.sys_status_label.setText(f"Downloading {model}...")
            self.sys_status_label.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 14px; font-weight: 500;")
        elif process == "translator" and status == "model_fallback":
            model = item.get("model", "the translation model")
            logger.warning(f"[UI] Translation model fallback active: {model}")
            self.sys_status_label.setText(f"Using fallback model {model}")
            self.sys_status_label.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 14px; font-weight: 500;")

    def _handle_partial(self, item):
        text = item.get("text", "")
        if text:
            self.current_label.show()
            self.current_label.setText(text)
            self.current_label.setStyleSheet("color: #CBD5E1; font-size: 18px; padding: 16px; background-color: #1E293B; border: 1px solid #334155; border-radius: 8px;")
            logger.debug(f"[UI] Partial transcript: '{text}'")

    def _handle_final(self, item):
        text = item.get("text", "")
        if text:
            self.current_label.show()
            self.current_label.setText(f"<i>{text}</i>")
            self.current_label.setStyleSheet("color: #94A3B8; font-style: italic; font-size: 18px; padding: 16px; background-color: #1E293B; border: 1px solid #334155; border-radius: 8px;")
            logger.info(f"[UI] Final transcript ({len(text)} chars): '{text}'")

    def _handle_provisional(self, item):
        # Live translation preview while recording continues.
        # Not added to history/log and never resets UI state —
        # the authoritative 'translation' event replaces it.
        # Guard: discard late provisionals that arrive after
        # the UI has been reset (B3 fix).
        if not self.is_recording:
            return
        original = item.get("original", "")
        translated = item.get("translated", "")
        if translated:
            self.current_label.show()
            self.current_label.setText(f"{original}\n→ {translated}")
            self.current_label.setStyleSheet("color: #D1FAE5; font-size: 18px; padding: 16px; background-color: #064E3B; border: 1px solid #047857; border-radius: 8px;")
            logger.debug(f"[UI] Provisional translation: '{translated}'")

    def _handle_translation(self, item):
        ui_display_time = time.time()
        original = item.get("original", "")
        translated = item.get("translated", "")
        latency = item.get("latency", 0.0)
        timing = item.get("timing", {})

        # Log full pipeline timing if available
        if timing.get("recording_start") and timing.get("recording_stop"):
            rec_dur, trans_dur, tl_dur, total = _pipeline_timing_values(timing, ui_display_time)
            status_tag = "[OK]" if total <= 2.0 else "[SLOW]"
            logger.info(
                f"[PIPELINE] Recording: {rec_dur:.2f}s | "
                f"Transcription: {trans_dur:.3f}s | "
                f"Translation: {tl_dur:.3f}s | "
                f"TOTAL (stop->display): {total:.3f}s {status_tag}"
            )

        self._add_to_history(original, translated, latency)
        self._log_message(original, translated, timing)
        logger.info(f"[UI] Translation displayed ({len(translated)} chars): '{translated}'")
        if self.is_recording:
            # Auto-committed segment mid-recording: show the translation
            # in the preview label but keep the recording state (button,
            # current transcript) untouched — the next partial/final from
            # the new segment will overwrite it.
            self.current_label.show()
            self.current_label.setText(f"{original}\n→ {translated}")
            self.current_label.setStyleSheet(
                "color: #D1FAE5; font-size: 18px; padding: 16px; "
                "background-color: #064E3B; border: 1px solid #047857; border-radius: 8px;"
            )
            return
        self._reset_ui_state()

    def _handle_cancel(self, item):
        reason = item.get("reason", "unknown")
        logger.info(f"[UI] Recording cancelled (reason={reason}).")
        if reason == "no_speech":
            # Genuine silence (VAD removed everything) — give the
            # user actionable feedback instead of a silent reset.
            self.current_label.show()
            self.current_label.setText(
                "<b>No speech detected.</b> Check that audio is "
                "reaching the selected device (e.g. Stereo Mix / "
                "virtual cable) and that something is playing."
            )
            self.current_label.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 15px; padding: 16px; background-color: #451A03; border: 1px solid #92400E; border-radius: 8px;")
        elif reason == "empty_audio":
            self.current_label.show()
            self.current_label.setText(
                "<b>Empty recording.</b> No audio was captured — "
                "check the microphone or virtual device."
            )
            self.current_label.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 15px; padding: 16px; background-color: #450A0A; border: 1px solid #7F1D1D; border-radius: 8px;")
        self._reset_ui_state()

    def _handle_skipped(self, item):
        reason = item.get("reason", "unknown")
        logger.info(f"[UI] Translation skipped (reason={reason}).")
        # Never reset the UI mid-recording: a late skip from a
        # previous utterance must not flip the button while the
        # user is already capturing the next one.
        if not self.is_recording:
            self._reset_ui_state()

    def _handle_truncated(self, item):
        dropped = item.get("dropped_seconds", 0)
        max_min = item.get("max_minutes", MAX_RECORDING_MINUTES_DEFAULT)
        logger.warning(f"[UI] Audio truncated at {max_min} min — dropped {dropped}s.")
        self.current_label.show()
        detail = (
            f" oldest {dropped:.1f}s dropped."
            if dropped > 0
            else " audio beyond this point is being discarded."
        )
        self.current_label.setText(
            f"<b>Warning:</b> recording truncated at {max_min} min —{detail}"
        )
        self.current_label.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 15px; padding: 16px; background-color: #451A03; border: 1px solid #92400E; border-radius: 8px;")

    def _handle_error(self, item):
        err_msg = item.get("message", "Unknown error")
        logger.error(f"[UI] Pipeline error received: {err_msg}")
        self.current_label.show()
        self.current_label.setText(f"<b>{err_msg}</b>")
        self.current_label.setStyleSheet("color: #F87171; font-size: 16px; padding: 16px; background-color: #450A0A; border: 1px solid #7F1D1D; border-radius: 8px;")
        self.sys_status_label.setText("System Error")
        self.sys_status_label.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 14px; font-weight: 500;")
        self._reset_ui_state(is_error=True)
        self.is_recording = False
        try:
            self.control_queue.put("FINISH", block=False)
        except Exception as e:
            logger.warning(f"Error sending FINISH to control_queue: {e}")

    def closeEvent(self, event):
        # Signal the watchdog to stand down — processes are being stopped
        # intentionally, so a dead worker here is NOT a crash.
        self._shutting_down = True
        self.stop_callback()
        event.accept()

def run_ui(ui_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, start_callback, stop_callback, log_path: str = None, health_check=None):
    app = QApplication(sys.argv)
    start_callback()
    window = MainWindow(ui_queue, control_queue, stop_callback, log_path, health_check=health_check)
    window.show()
    sys.exit(app.exec())
