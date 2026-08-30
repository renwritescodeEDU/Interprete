import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import queue

# Set offscreen platform before importing Qt
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt6.QtWidgets import QApplication, QLabel
from src.ui import _load_config, _save_config, MainWindow, run_ui

# Create a global QApplication instance for tests
_app = QApplication.instance()
if not _app:
    _app = QApplication(sys.argv)

class TestUI(unittest.TestCase):
    def test_ui_callable(self):
        """test_ui_callable - verify run_ui is a callable function."""
        self.assertTrue(callable(run_ui))

    @patch('src.ui.QApplication')
    @patch('src.ui.MainWindow')
    @patch('src.ui.sys.exit')
    def test_run_ui(self, mock_exit, mock_main_window, mock_qapp):
        """test_run_ui - verify run_ui initializes components and calls callbacks."""
        mock_app_inst = MagicMock()
        mock_qapp.return_value = mock_app_inst
        
        ui_queue = MagicMock()
        control_queue = MagicMock()
        start_callback = MagicMock()
        stop_callback = MagicMock()
        
        run_ui(ui_queue, control_queue, start_callback, stop_callback, "log.txt")
        
        start_callback.assert_called_once()
        mock_main_window.assert_called_once_with(ui_queue, control_queue, stop_callback, "log.txt", health_check=None)
        mock_app_inst.exec.assert_called_once()
        mock_exit.assert_called_once()

    def test_ui_classes_exist(self):
        """test_ui_classes_exist - verify MainWindow has the required methods."""
        self.assertTrue(hasattr(MainWindow, 'poll_queue'))
        self.assertTrue(hasattr(MainWindow, 'on_action_clicked'))
        self.assertTrue(hasattr(MainWindow, 'update_system_readiness'))
        self.assertTrue(hasattr(MainWindow, '_add_to_history'))
        self.assertTrue(hasattr(MainWindow, '_log_message'))
        self.assertTrue(hasattr(MainWindow, '_reset_ui_state'))

    @patch('src.config.os.path.exists')
    def test_load_config_missing_file(self, mock_exists):
        """test_load_config_missing_file - verify returns empty dict when file doesn't exist."""
        mock_exists.return_value = False
        config = _load_config()
        self.assertEqual(config, {})

    def test_save_load_config_roundtrip(self):
        """test_save_load_config_roundtrip - save then load config, verify data is preserved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = os.path.join(tmp_dir, "preferences.json")
            test_config = {"test_key": "test_value", "number": 42}

            with patch('src.ui.CONFIG_FILE', config_file):
                _save_config(test_config)
                self.assertTrue(os.path.exists(config_file))
                loaded_config = _load_config()
                self.assertEqual(loaded_config, test_config)

    def test_log_message_basic(self):
        """test_log_message_basic - verify _log_message writes to the log path."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_path = tmp_file.name
        
        try:
            with patch('src.ui.MainWindow._setup_ui'), \
                 patch('src.ui.MainWindow._center_on_screen'), \
                 patch('src.ui.MainWindow._refresh_devices'), \
                 patch('src.ui.MainWindow._restore_saved_device'), \
                 patch('src.ui.QTimer'):
                
                window = MainWindow(MagicMock(), MagicMock(), MagicMock(), log_path)
                window._log_message("orig_text", "trans_text")
                
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertIn("Original: orig_text", content)
                    self.assertIn("Traducido: trans_text", content)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_log_message_with_timing(self):
        """test_log_message_with_timing - verify timing data is logged correctly."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_path = tmp_file.name
        
        try:
            with patch('src.ui.MainWindow._setup_ui'), \
                 patch('src.ui.MainWindow._center_on_screen'), \
                 patch('src.ui.MainWindow._refresh_devices'), \
                 patch('src.ui.MainWindow._restore_saved_device'), \
                 patch('src.ui.QTimer'):
                
                window = MainWindow(MagicMock(), MagicMock(), MagicMock(), log_path)
                timing = {
                    "recording_start": 100.0,
                    "recording_stop": 102.0,
                    "transcription_start": 102.1,
                    "transcription_end": 102.5,
                    "translation_start": 102.5,
                    "translation_end": 103.0
                }
                
                with patch('src.ui.time.time', return_value=103.5):
                    window._log_message("orig_test", "trans_test", timing)
                
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertIn("rec=2.00s", content)
                    self.assertIn("asr=0.400s", content)
                    self.assertIn("tl=0.500s", content)
                    self.assertIn("total=1.500s", content)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    @patch('src.ui.open')
    def test_log_message_no_path(self, mock_open):
        """test_log_message_no_path - verify nothing is written if log_path is None."""
        with patch('src.ui.MainWindow._setup_ui'), \
             patch('src.ui.MainWindow._center_on_screen'), \
             patch('src.ui.MainWindow._refresh_devices'), \
             patch('src.ui.MainWindow._restore_saved_device'), \
             patch('src.ui.QTimer'):
            
            window = MainWindow(MagicMock(), MagicMock(), MagicMock(), None)
            window._log_message("orig", "trans")
            mock_open.assert_not_called()

class TestMainWindow(unittest.TestCase):
    def setUp(self):
        self.ui_queue = MagicMock()
        self.control_queue = MagicMock()
        self.stop_callback = MagicMock()
        self.log_path = None
        
        self.patcher1 = patch('src.ui.list_audio_devices', return_value=[])
        self.patcher2 = patch('src.ui.QGuiApplication.primaryScreen', return_value=MagicMock())
        self.patcher3 = patch('src.ui.QTimer.start')
        
        self.mock_list_devices = self.patcher1.start()
        self.mock_primary_screen = self.patcher2.start()
        self.mock_timer_start = self.patcher3.start()
        
        self.window = MainWindow(self.ui_queue, self.control_queue, self.stop_callback, self.log_path)
    
    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.window.close()

    def test_poll_queue_status_transcriber_ready(self):
        """test_poll_queue_status_transcriber_ready - handle transcriber ready message."""
        self.ui_queue.get_nowait.side_effect = [{"type": "status", "process": "transcriber", "status": "ready"}, queue.Empty]
        self.window.poll_queue()
        self.assertTrue(self.window.transcriber_ready)
        self.assertFalse(self.window.translator_ready)
        
    def test_poll_queue_status_both_ready(self):
        """test_poll_queue_status_both_ready - handle transcriber, translator, and audio ready messages."""
        self.ui_queue.get_nowait.side_effect = [
            {"type": "status", "process": "audio", "status": "ready"},
            {"type": "status", "process": "transcriber", "status": "ready"},
            {"type": "status", "process": "translator", "status": "ready"},
            queue.Empty
        ]
        self.window.poll_queue()
        self.assertTrue(self.window.transcriber_ready)
        self.assertTrue(self.window.translator_ready)
        self.assertTrue(self.window.audio_ready)
        self.assertTrue(self.window.action_btn.isEnabled())
        self.assertEqual(self.window.action_btn.text(), "Start Recording")
        
    @patch('src.ui.MainWindow._add_to_history')
    @patch('src.ui.MainWindow._log_message')
    def test_poll_queue_translation_with_timing(self, mock_log_message, mock_add_to_history):
        """test_poll_queue_translation_with_timing - process translation queue item and verify history/log calls."""
        timing = {"recording_start": 100.0}
        self.ui_queue.get_nowait.side_effect = [
            {"type": "translation", "original": "hello", "translated": "hola", "latency": 0.5, "timing": timing},
            queue.Empty
        ]
        self.window.poll_queue()
        mock_add_to_history.assert_called_once_with("hello", "hola", 0.5)
        mock_log_message.assert_called_once_with("hello", "hola", timing)
        
    @patch('src.ui.MainWindow._reset_ui_state')
    def test_poll_queue_cancel(self, mock_reset):
        """test_poll_queue_cancel - verify cancel queue item resets UI state."""
        self.ui_queue.get_nowait.side_effect = [{"type": "cancel"}, queue.Empty]
        self.window.poll_queue()
        mock_reset.assert_called_once()

    @patch('src.ui.MainWindow._reset_ui_state')
    def test_poll_queue_skipped(self, mock_reset):
        """test_poll_queue_skipped - a skipped terminal event resets the UI (no lock)."""
        self.ui_queue.get_nowait.side_effect = [
            {"type": "skipped", "reason": "same_language"}, queue.Empty
        ]
        self.window.poll_queue()
        mock_reset.assert_called_once()

    @patch('src.ui.MainWindow._reset_ui_state')
    def test_poll_queue_skipped_while_recording(self, mock_reset):
        """test_poll_queue_skipped_while_recording - a late skip must not reset the UI mid-recording."""
        self.window.is_recording = True
        self.ui_queue.get_nowait.side_effect = [
            {"type": "skipped", "reason": "same_language"}, queue.Empty
        ]
        self.window.poll_queue()
        mock_reset.assert_not_called()

    def test_poll_queue_truncated(self):
        """test_poll_queue_truncated - truncated event warns without resetting state."""
        with patch.object(self.window, '_reset_ui_state') as mock_reset:
            self.ui_queue.get_nowait.side_effect = [
                {"type": "truncated", "dropped_seconds": 12.5, "max_minutes": 5},
                queue.Empty
            ]
            self.window.poll_queue()
        mock_reset.assert_not_called()
        self.assertIn("12.5", self.window.current_label.text())
        self.assertIn("Warning", self.window.current_label.text())

    def test_poll_queue_provisional(self):
        """test_poll_queue_provisional - while recording, provisional preview
        shows translation without resetting UI state."""
        self.window.is_recording = True
        with patch.object(self.window, '_reset_ui_state') as mock_reset:
            self.ui_queue.get_nowait.side_effect = [
                {"type": "provisional", "original": "Buenos días", "translated": "Good morning"},
                queue.Empty
            ]
            self.window.poll_queue()
        mock_reset.assert_not_called()
        self.assertIn("Good morning", self.window.current_label.text())

    def test_poll_queue_error(self):
        """test_poll_queue_error - verify error queue item displays the error."""
        self.ui_queue.get_nowait.side_effect = [{"type": "error", "message": "API Failure"}, queue.Empty]
        self.window.poll_queue()
        self.assertIn("API Failure", self.window.current_label.text())
        self.assertEqual(self.window.sys_status_label.text(), "System Error")
        self.assertFalse(self.window.is_recording)
        
    def test_on_action_clicked_start(self):
        """test_on_action_clicked_start - verify clicking the action button starts recording."""
        self.window.is_recording = False
        
        with patch('src.ui.time.time', return_value=500.0):
            self.window.on_action_clicked()
            
        self.assertTrue(self.window.is_recording)
        self.assertEqual(self.window._recording_start_time, 500.0)
        self.assertEqual(self.window.sys_status_label.text(), "Recording...")
        self.control_queue.put.assert_called_once_with(("START", 500.0), block=False)
        
    def test_on_action_clicked_stop(self):
        """test_on_action_clicked_stop - verify clicking the action button stops recording."""
        self.window.is_recording = True
        self.window._recording_start_time = 500.0
        
        with patch('src.ui.time.time', return_value=510.0):
            self.window.on_action_clicked()
            
        self.assertFalse(self.window.is_recording)
        self.assertEqual(self.window.sys_status_label.text(), "Translating...")
        self.control_queue.put.assert_called_once_with(("FINISH", 510.0), block=False)
        
    def test_add_to_history_renders_clean_text(self):
        """test_add_to_history_renders_clean_text - verify bubble contains exact translated text without model prefixes."""
        initial_count = self.window.history_layout.count()
        self.window._add_to_history("Hello", "Hola", latency=1.23)
        self.assertEqual(self.window.history_layout.count(), initial_count + 1)
        
        # Check the newly added bubble item
        item = self.window.history_layout.itemAt(initial_count)
        bubble = item.widget()
        labels = bubble.findChildren(QLabel)
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0].text(), "Hello")
        self.assertEqual(labels[1].text(), "Hola")

    def test_add_to_history_capped(self):
        """test_add_to_history_capped - history must not grow without bound."""
        for i in range(110):
            self.window._add_to_history(f"orig{i}", f"trans{i}", latency=0.1)
        self.assertEqual(self.window.history_layout.count(), 100)
        # Oldest entry was evicted, newest remains
        newest = self.window.history_layout.itemAt(self.window.history_layout.count() - 1)
        labels = newest.widget().findChildren(QLabel)
        self.assertEqual(labels[1].text(), "trans109")

    def test_check_health_all_alive(self):
        """test_check_health_all_alive - healthy workers leave UI untouched."""
        self.window.transcriber_ready = True
        self.window.translator_ready = True
        self.window.health_check = lambda: {"audio": True, "transcriber": True, "translator": True}
        self.window.check_health()
        self.assertFalse(self.window._workers_dead)
        self.assertNotEqual(self.window.sys_status_label.text(), "System Error")

    def test_check_health_reports_dead_workers(self):
        """test_check_health_reports_dead_workers - dead workers surface an error."""
        self.window.transcriber_ready = True
        self.window.translator_ready = True
        self.window.health_check = lambda: {"audio": True, "transcriber": False, "translator": True}
        self.window.check_health()  # 1st miss
        self.window.check_health()  # 2nd consecutive miss → crash declared
        self.assertTrue(self.window._workers_dead)
        self.assertIn("Worker crash", self.window.current_label.text())
        self.assertIn("transcriber", self.window.current_label.text())
        self.assertEqual(self.window.sys_status_label.text(), "System Error")
        self.assertFalse(self.window.action_btn.isEnabled())

    def test_check_health_not_ready(self):
        """test_check_health_not_ready - watchdog stays silent before workers are ready."""
        self.window.health_check = lambda: {"audio": False, "transcriber": False, "translator": False}
        self.window.check_health()
        self.assertFalse(self.window._workers_dead)

    def test_check_health_requires_consecutive_misses(self):
        """test_check_health_requires_consecutive_misses - a single miss is not a crash."""
        self.window.transcriber_ready = True
        self.window.translator_ready = True
        self.window.health_check = lambda: {"audio": True, "transcriber": False, "translator": True}
        self.window.check_health()  # 1st miss
        self.assertFalse(self.window._workers_dead)
        self.assertNotEqual(self.window.sys_status_label.text(), "System Error")
        self.window.check_health()  # 2nd consecutive miss
        self.assertTrue(self.window._workers_dead)

    def test_check_health_silent_during_shutdown(self):
        """test_check_health_silent_during_shutdown - watchdog stands down on close."""
        self.window.transcriber_ready = True
        self.window.translator_ready = True
        self.window.health_check = lambda: {"audio": False, "transcriber": False, "translator": False}
        self.window._shutting_down = True
        self.window.check_health()
        self.assertFalse(self.window._workers_dead)
        self.assertNotEqual(self.window.sys_status_label.text(), "System Error")

    @patch('src.ui.MainWindow._add_to_history')
    @patch('src.ui.MainWindow._log_message')
    @patch('src.ui.MainWindow._reset_ui_state')
    def test_poll_queue_logs_pipeline_timing(self, mock_reset, mock_log_message, mock_add_to_history):
        """The stop->display pipeline timing line must use the exact current format."""
        timing = {
            "recording_start": 100.0,
            "recording_stop": 102.0,
            "transcription_start": 102.1,
            "transcription_end": 102.5,
            "translation_start": 102.5,
            "translation_end": 103.0,
        }
        self.ui_queue.get_nowait.side_effect = [
            {"type": "translation", "original": "hello", "translated": "hola",
             "latency": 0.5, "timing": timing},
            queue.Empty,
        ]
        with patch('src.ui.time.time', return_value=103.5), \
             patch('src.ui.logger.info') as mock_log_info:
            self.window.poll_queue()
        pipeline = [c.args[0] for c in mock_log_info.call_args_list
                    if c.args and str(c.args[0]).startswith("[PIPELINE]")]
        self.assertEqual(len(pipeline), 1)
        self.assertIn("Recording: 2.00s", pipeline[0])
        self.assertIn("Transcription: 0.400s", pipeline[0])
        self.assertIn("Translation: 0.500s", pipeline[0])
        self.assertIn("TOTAL (stop->display): 1.500s [OK]", pipeline[0])

    def test_pipeline_timing_slow_tag(self):
        """Total beyond the 2s budget must be flagged [SLOW]."""
        timing = {
            "recording_start": 100.0,
            "recording_stop": 102.0,
            "transcription_start": 102.1,
            "transcription_end": 102.5,
            "translation_start": 102.5,
            "translation_end": 103.0,
        }
        self.ui_queue.get_nowait.side_effect = [
            {"type": "translation", "original": "hello", "translated": "hola",
             "latency": 0.5, "timing": timing},
            queue.Empty,
        ]
        with patch('src.ui.time.time', return_value=105.0), \
             patch('src.ui.logger.info') as mock_log_info:
            self.window.poll_queue()
        pipeline = [c.args[0] for c in mock_log_info.call_args_list
                    if c.args and str(c.args[0]).startswith("[PIPELINE]")]
        self.assertEqual(len(pipeline), 1)
        self.assertIn("TOTAL (stop->display): 3.000s [SLOW]", pipeline[0])

    def test_late_provisional_does_not_override_reset(self):
        """A provisional arriving after the UI has been reset (is_recording=False)
        must NOT update current_label. This test FAILS on the current code (B3 bug)."""
        self.window._reset_ui_state()
        self.window.is_recording = False
        self.window.current_label.hide()
        self.window.current_label.setText("")

        self.ui_queue.get_nowait.side_effect = [
            {"type": "provisional", "original": "Buenos días", "translated": "Good morning"},
            queue.Empty,
        ]
        self.window.poll_queue()
        self.assertEqual(
            self.window.current_label.text(), "",
            "B3 BUG: provisional updated current_label even though is_recording=False"
        )

    def test_poll_queue_status_model_fallback(self):
        """model_fallback status must be surfaced in the system status label."""
        self.ui_queue.get_nowait.side_effect = [
            {"type": "status", "process": "translator", "status": "model_fallback", "model": "llama3.2:3b"},
            queue.Empty,
        ]
        self.window.poll_queue()
        self.assertIn("llama3.2:3b", self.window.sys_status_label.text())
        self.assertIn("fallback", self.window.sys_status_label.text())

    def test_translation_while_recording_keeps_button(self):
        """A translation event arriving mid-recording (auto-commit) must NOT
        reset the UI state — the button must stay 'Stop Recording', the
        translation must appear in current_label, and the bubble must be
        created in the history panel (not via _add_to_history)."""
        self.window.is_recording = True
        self.window.action_btn.setText("Stop Recording")
        self.window.current_label.show()
        self.window.current_label.setText("Original transcript...")
        initial_count = self.window.history_layout.count()

        with patch.object(self.window, '_reset_ui_state') as mock_reset, \
             patch.object(self.window, '_add_to_history') as mock_history, \
             patch.object(self.window, '_log_message') as mock_log:
            self.ui_queue.get_nowait.side_effect = [
                {"type": "translation", "original": "hello world",
                 "translated": "hola mundo", "latency": 1.5, "timing": {}},
                queue.Empty,
            ]
            self.window.poll_queue()

        mock_reset.assert_not_called()
        mock_history.assert_not_called()
        mock_log.assert_called_once()
        self.assertEqual(self.window.action_btn.text(), "Stop Recording",
                         "mid-recording translation must not flip the button to Start")
        self.assertIn("hola mundo", self.window.current_label.text())
        # The bubble must be created in the history panel
        self.assertEqual(self.window.history_layout.count(), initial_count + 1)

    def test_translation_appends_to_live_bubble(self):
        """Two auto-commit fragments mid-recording must concatenate into ONE
        history bubble (space-separated), not two separate bubbles."""
        self.window.is_recording = True
        self.window.action_btn.setText("Stop Recording")
        initial_count = self.window.history_layout.count()

        self.ui_queue.get_nowait.side_effect = [
            {"type": "translation", "original": "hello world",
             "translated": "hola mundo", "latency": 1.5, "timing": {}},
            {"type": "translation", "original": "how are you",
             "translated": "cómo estás", "latency": 1.2, "timing": {}},
            queue.Empty,
        ]
        self.window.poll_queue()

        self.assertEqual(self.window.history_layout.count(), initial_count + 1,
                         "must be exactly one bubble, not two")
        item = self.window.history_layout.itemAt(initial_count)
        labels = item.widget().findChildren(QLabel)
        orig_text = labels[0].text()
        trans_text = labels[1].text()
        self.assertEqual(orig_text, "hello world how are you")
        self.assertEqual(trans_text, "hola mundo cómo estás")

    def test_translation_after_stop_continues_live_bubble(self):
        """A mid-recording fragment followed by a STOP fragment must produce
        ONE bubble containing both fragments."""
        self.window.is_recording = True
        self.window.action_btn.setText("Stop Recording")
        initial_count = self.window.history_layout.count()

        self.ui_queue.get_nowait.side_effect = [
            {"type": "translation", "original": "hello world",
             "translated": "hola mundo", "latency": 1.5, "timing": {}},
            queue.Empty,
        ]
        self.window.poll_queue()
        self.assertEqual(self.window.history_layout.count(), initial_count + 1)

        # Now STOP: the final fragment must append to the same bubble
        self.window.is_recording = False
        self.ui_queue.get_nowait.side_effect = [
            {"type": "translation", "original": "how are you",
             "translated": "cómo estás", "latency": 1.2, "timing": {}},
            queue.Empty,
        ]
        self.window.poll_queue()

        self.assertEqual(self.window.history_layout.count(), initial_count + 1,
                         "STOP must not create a second bubble — append to live one")
        item = self.window.history_layout.itemAt(initial_count)
        labels = item.widget().findChildren(QLabel)
        self.assertEqual(labels[0].text(), "hello world how are you")
        self.assertEqual(labels[1].text(), "hola mundo cómo estás")

    def test_translation_after_stop_fresh_bubble(self):
        """Without any mid-recording auto-commits, a STOP translation creates
        a fresh bubble via _add_to_history (existing behaviour preserved)."""
        self.window.is_recording = False
        with patch.object(self.window, '_add_to_history') as mock_history:
            self.ui_queue.get_nowait.side_effect = [
                {"type": "translation", "original": "hello world",
                 "translated": "hola mundo", "latency": 1.5, "timing": {}},
                queue.Empty,
            ]
            self.window.poll_queue()
        mock_history.assert_called_once_with("hello world", "hola mundo", 1.5)


if __name__ == '__main__':
    unittest.main()
