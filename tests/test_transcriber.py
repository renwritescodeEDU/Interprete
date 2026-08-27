import queue
import unittest
from unittest.mock import MagicMock, patch

from src.transcriber import _send_to_queue, start_transcriber

class TestTranscriber(unittest.TestCase):
    def setUp(self):
        self.asr_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.ui_queue = queue.Queue()

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_callable(self, mock_whisper):
        """Test that start_transcriber is callable and exits cleanly."""
        self.asr_queue.put("QUIT")
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        self.assertTrue(mock_whisper.called)

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_status_ready(self, mock_whisper):
        """Verify the first message in ui_queue is status ready."""
        self.asr_queue.put("QUIT")
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        msg = self.ui_queue.get_nowait()
        self.assertEqual(msg, {"type": "status", "process": "transcriber", "status": "ready"})

    @patch("src.transcriber.time.time", side_effect=[100.0, 101.0])
    @patch("src.transcriber.WhisperModel")
    def test_transcriber_processes_queue(self, mock_whisper, mock_time):
        """Send audio+poison pill, verify 3-element tuple in translation_queue with timing."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        timing = {"capture_start": 99.0}
        self.asr_queue.put((b"audio_data", 16000, True, timing))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        msg = self.translation_queue.get_nowait()
        self.assertEqual(len(msg), 3)
        text, lang, out_timing = msg
        self.assertEqual(text, "Hello world")
        self.assertEqual(lang, "en")
        self.assertEqual(out_timing["capture_start"], 99.0)
        self.assertEqual(out_timing["transcription_start"], 100.0)
        self.assertEqual(out_timing["transcription_end"], 101.0)

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_empty_audio(self, mock_whisper):
        """Send empty audio with is_final=True, verify cancel sent to ui_queue."""
        self.asr_queue.put((b"", 16000, True))
        self.asr_queue.put("QUIT")
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Skip 'ready' message
        self.ui_queue.get_nowait()
        cancel_msg = self.ui_queue.get_nowait()
        self.assertEqual(cancel_msg, {"type": "cancel"})

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_unsupported_language(self, mock_whisper):
        """Mock language='fr', verify cancel and no output in translation_queue."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Bonjour"
        mock_info = MagicMock()
        mock_info.language = "fr"
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        self.asr_queue.put((b"audio_data", 16000, True))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Skip 'ready' message
        self.ui_queue.get_nowait()
        cancel_msg = self.ui_queue.get_nowait()
        self.assertEqual(cancel_msg, {"type": "cancel"})
        self.assertTrue(self.translation_queue.empty())

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_partial_transcript(self, mock_whisper):
        """is_final=False, verify partial message in ui_queue."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello "
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        self.asr_queue.put((b"audio_data", 16000, False))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Skip 'ready' message
        self.ui_queue.get_nowait()
        partial_msg = self.ui_queue.get_nowait()
        self.assertEqual(partial_msg, {"type": "partial", "text": "Hello"})
        self.assertTrue(self.translation_queue.empty())

    @patch("src.transcriber.time.time", side_effect=[100.0, 101.0])
    @patch("src.transcriber.WhisperModel")
    def test_transcriber_timing_propagation(self, mock_whisper, mock_time):
        """Send 4-element tuple with timing, verify timing dict updated."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        timing = {"capture": 1.0}
        self.asr_queue.put((b"audio", 16000, True, timing))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        msg = self.translation_queue.get_nowait()
        out_timing = msg[2]
        self.assertEqual(out_timing["capture"], 1.0)
        self.assertEqual(out_timing["transcription_start"], 100.0)
        self.assertEqual(out_timing["transcription_end"], 101.0)

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_legacy_3element_tuple(self, mock_whisper):
        """Send 3-element tuple (no timing), verify still works."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        self.asr_queue.put((b"audio", 16000, True))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        msg = self.translation_queue.get_nowait()
        self.assertEqual(len(msg), 3)
        self.assertEqual(msg[0], "Hello")
        self.assertEqual(msg[1], "en")
        self.assertIn("transcription_start", msg[2])

    def test_send_to_queue_full(self):
        """Test _send_to_queue with a full queue (maxsize=1)."""
        q = queue.Queue(maxsize=1)
        q.put("first")
        
        with patch("src.transcriber.logger.error") as mock_error:
            _send_to_queue(q, "second", block=True, timeout=0.01, error_msg="Queue is full")
            mock_error.assert_called_once_with("Queue is full")

    def test_send_to_queue_exception(self):
        """Test _send_to_queue general exception."""
        mock_queue = MagicMock()
        mock_queue.put.side_effect = Exception("General error")
        
        with patch("src.transcriber.logger.debug") as mock_debug:
            _send_to_queue(mock_queue, "message")
            mock_debug.assert_called_once()
            self.assertIn("Queue communication error", mock_debug.call_args[0][0])

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_error_handling(self, mock_whisper):
        """Mock model.transcribe to raise, verify error in ui_queue."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        mock_model.transcribe.side_effect = RuntimeError("Whisper failed")
        
        self.asr_queue.put((b"audio", 16000, True))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Skip 'ready' message
        self.ui_queue.get_nowait()
        error_msg = self.ui_queue.get_nowait()
        self.assertEqual(error_msg["type"], "error")
        self.assertIn("Whisper failed", error_msg["message"])

if __name__ == "__main__":
    unittest.main()
