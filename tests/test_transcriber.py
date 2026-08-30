import queue
import unittest
from unittest.mock import MagicMock, patch

from src.transcriber import TRANSCRIBE_INITIAL_PROMPT, _send_to_queue, start_transcriber

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

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_processes_queue(self, mock_whisper):
        """Send audio+poison pill, verify 3-element tuple in translation_queue with timing."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
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
        self.assertIn("transcription_start", out_timing)
        self.assertIn("transcription_end", out_timing)
        self.assertGreaterEqual(out_timing["transcription_end"], out_timing["transcription_start"])

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_empty_audio(self, mock_whisper):
        """Send empty audio with is_final=True, verify cancel sent to ui_queue."""
        self.asr_queue.put((b"", 16000, True))
        self.asr_queue.put("QUIT")
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Skip 'ready' message
        self.ui_queue.get_nowait()
        cancel_msg = self.ui_queue.get_nowait()
        self.assertEqual(cancel_msg, {"type": "cancel", "reason": "empty_audio"})

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_unsupported_language_falls_back(self, mock_whisper):
        """Mock language='fr': system falls back to DEFAULT_LANGUAGE instead of cancelling."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Bonjour"
        mock_info = MagicMock()
        mock_info.language = "fr"
        mock_info.language_probability = 0.90
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        self.asr_queue.put((b"audio_data", 16000, True))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Skip 'ready' message
        self.ui_queue.get_nowait()
        # No cancel: the unsupported language falls back to 'en' and the
        # transcript is forwarded for translation instead of being discarded.
        final_msg = self.ui_queue.get_nowait()
        self.assertEqual(final_msg["type"], "final")
        msg = self.translation_queue.get_nowait()
        text, lang, _ = msg
        self.assertEqual(text, "Bonjour")
        self.assertEqual(lang, "en")

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_low_confidence_final_still_forwarded(self, mock_whisper):
        """Final with low language probability is transcribed, not discarded."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.40
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        self.asr_queue.put((b"audio_data", 16000, True))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Skip 'ready' message
        self.ui_queue.get_nowait()
        final_msg = self.ui_queue.get_nowait()
        self.assertEqual(final_msg["type"], "final")
        msg = self.translation_queue.get_nowait()
        text, lang, _ = msg
        self.assertEqual(text, "Hello world")
        self.assertEqual(lang, "en")

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_partial_transcript(self, mock_whisper):
        """is_final=False, verify partial message in ui_queue."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello "
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.85
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        self.asr_queue.put((b"audio_data", 16000, False))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Skip 'ready' message
        self.ui_queue.get_nowait()
        partial_msg = self.ui_queue.get_nowait()
        self.assertEqual(partial_msg, {"type": "partial", "text": "Hello"})
        self.assertTrue(self.translation_queue.empty())

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_timing_propagation(self, mock_whisper):
        """Send 4-element tuple with timing, verify timing dict updated."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        timing = {"capture": 1.0}
        self.asr_queue.put((b"audio", 16000, True, timing))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        msg = self.translation_queue.get_nowait()
        out_timing = msg[2]
        self.assertEqual(out_timing["capture"], 1.0)
        self.assertIn("transcription_start", out_timing)
        self.assertIn("transcription_end", out_timing)
        self.assertGreaterEqual(out_timing["transcription_end"], out_timing["transcription_start"])

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_legacy_3element_tuple(self, mock_whisper):
        """Send 3-element tuple (no timing), verify still works."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
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
        
        with patch("src.queueutil.logger.error") as mock_error:
            _send_to_queue(q, "second", block=True, timeout=0.01, error_msg="Queue is full")
            mock_error.assert_called_once_with("Queue is full")

    def test_send_to_queue_exception(self):
        """Test _send_to_queue general exception."""
        mock_queue = MagicMock()
        mock_queue.put.side_effect = Exception("General error")
        
        with patch("src.queueutil.logger.debug") as mock_debug:
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

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_partial_sends_provisional_task(self, mock_whisper):
        """Accumulated partials beyond the growth threshold are queued as provisional tasks."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "This is a longer test utterance"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.90
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        self.asr_queue.put((b"audio", 16000, False))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        # Partial (30 chars) exceeds PARTIAL_PROGRESS_THRESHOLD → 4-tuple with is_partial=True
        msg = self.translation_queue.get_nowait()
        self.assertEqual(len(msg), 4)
        text, lang, timing, is_partial = msg
        self.assertEqual(text, "This is a longer test utterance")
        self.assertEqual(lang, "en")
        self.assertTrue(is_partial)

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_partial_below_threshold_not_queued(self, mock_whisper):
        """Tiny partials under the growth threshold are not sent to the translator."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hola"  # 4 chars < threshold
        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.90
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        self.asr_queue.put((b"audio", 16000, False))
        self.asr_queue.put("QUIT")
        
        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        self.assertTrue(self.translation_queue.empty())

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_priority_final_queue(self, mock_whisper):
        """A final on the dedicated priority queue is processed before queued partials."""
        import queue as queue_mod
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        final_queue = queue_mod.Queue()
        # Partial queued first on asr_queue; final arrives on the priority queue
        self.asr_queue.put((b"partial_audio", 16000, False))
        final_queue.put((b"final_audio", 16000, True, {"recording_stop": 1.0}))
        self.asr_queue.put("QUIT")

        start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue, final_queue)

        # The final is transcribed first and queued for translation as a 3-tuple
        msg = self.translation_queue.get_nowait()
        self.assertEqual(len(msg), 3)
        text, lang, timing = msg
        self.assertEqual(text, "Hello world")
        self.assertEqual(lang, "en")

    @patch("src.transcriber.WhisperModel")
    def test_transcriber_queue_full_emits_skipped(self, mock_whisper):
        """When translation_queue is full, a skipped terminal event must reach the UI."""
        from src import transcriber
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        real_send = transcriber._send_to_queue

        def fake_send(q, msg, block=False, timeout=None, error_msg="Queue put failed"):
            if q is self.translation_queue and isinstance(msg, tuple):
                return False
            return real_send(q, msg, block=block, timeout=timeout, error_msg=error_msg)
        
        self.asr_queue.put((b"audio", 16000, True, {}))
        self.asr_queue.put("QUIT")
        
        with patch.object(transcriber, "_send_to_queue", side_effect=fake_send):
            start_transcriber(self.asr_queue, self.translation_queue, self.ui_queue)
        
        messages = []
        while not self.ui_queue.empty():
            messages.append(self.ui_queue.get_nowait())
        
        skipped = [m for m in messages if m.get("type") == "skipped"]
        self.assertTrue(skipped, f"expected a skipped terminal event in {messages}")
        self.assertEqual(skipped[0]["reason"], "queue_full")
        self.assertEqual(skipped[0]["stage"], "translation")


class TestModelSelection(unittest.TestCase):
    """Hardware-aware whisper model selection (Phase 7)."""

    @patch("src.transcriber.select_whisper_config",
           return_value={"model": "medium", "compute_type": "float16", "device": "cuda"})
    @patch("src.transcriber.WhisperModel")
    def test_create_model_uses_resolved_config(self, mock_whisper, mock_select):
        """_create_model must instantiate WhisperModel with the resolved config."""
        from src.transcriber import _create_model
        _create_model()
        mock_whisper.assert_called_once_with(
            "medium", device="cuda", compute_type="float16"
        )

    @patch("src.transcriber.select_whisper_config",
           return_value={"model": "medium", "compute_type": "float16", "device": "cuda"})
    @patch("src.transcriber.WhisperModel")
    def test_create_model_degrades_to_cpu_on_cuda_failure(self, mock_whisper, mock_select):
        """CUDA creation failure must fall back to CPU small/int8, never crash."""
        mock_whisper.side_effect = [RuntimeError("CUDA init failed"), MagicMock()]
        from src.transcriber import _create_model
        _create_model()
        self.assertEqual(mock_whisper.call_count, 2)
        args, kwargs = mock_whisper.call_args
        self.assertEqual(kwargs.get("device"), "cpu")
        self.assertEqual(kwargs.get("compute_type"), "int8")


class TestBilingualPrompt(unittest.TestCase):
    """Bilingual initial prompt for transcriptions (Phase 8)."""

    @patch("src.transcriber.WhisperModel")
    def test_partial_transcribe_uses_bilingual_prompt(self, mock_whisper):
        """Partial chunks must receive the bilingual initial prompt with both
        English and Spanish punctuation examples."""
        from src.transcriber import TRANSCRIBE_INITIAL_PROMPT
        self.assertIn("Hello", TRANSCRIBE_INITIAL_PROMPT)
        self.assertIn("Hola", TRANSCRIBE_INITIAL_PROMPT)
        self.assertIn("¿", TRANSCRIBE_INITIAL_PROMPT)

        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        mock_seg = MagicMock()
        mock_seg.text = "Hello"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        from src.transcriber import start_transcriber
        asr_q = queue.Queue()
        tr_q = queue.Queue()
        ui_q = queue.Queue()
        asr_q.put((b"audio_data", 16000, False))
        asr_q.put("QUIT")
        start_transcriber(asr_q, tr_q, ui_q)

        call_kwargs = mock_model.transcribe.call_args
        kwargs = call_kwargs[1] if len(call_kwargs) > 1 else {}
        self.assertIn("initial_prompt", kwargs)
        self.assertEqual(kwargs["initial_prompt"], TRANSCRIBE_INITIAL_PROMPT)

    @patch("src.transcriber.WhisperModel")
    def test_final_transcribe_chains_previous_final(self, mock_whisper):
        """Final chunks must receive an initial prompt that includes the text
        of the previous final, so Whisper keeps the thread of the conversation
        (context chaining, Phase 8.2)."""
        import threading
        import time
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        mock_seg = MagicMock()
        mock_seg.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        from src.transcriber import start_transcriber
        asr_q = queue.Queue()
        tr_q = queue.Queue()
        ui_q = queue.Queue()

        t = threading.Thread(target=start_transcriber, args=(asr_q, tr_q, ui_q))
        t.start()

        def wait_for_calls(n, timeout=5.0):
            deadline = time.time() + timeout
            while time.time() < deadline:
                if len(mock_model.transcribe.call_args_list) >= n:
                    return
                time.sleep(0.05)
            raise AssertionError(
                f"expected {n} transcribe calls, got {len(mock_model.transcribe.call_args_list)}"
            )

        # Feed finals progressively: a final queued while the previous one is
        # still being processed is drained as a stale partial, so wait between.
        asr_q.put((b"audio_data_1", 16000, True))
        wait_for_calls(1)
        asr_q.put((b"audio_data_2", 16000, True))
        wait_for_calls(2)
        asr_q.put("QUIT")
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "transcriber thread did not exit")

        calls = mock_model.transcribe.call_args_list
        self.assertGreaterEqual(len(calls), 2)
        # First final: no prior context -> plain bilingual prompt.
        first_kwargs = calls[0][1]
        self.assertEqual(first_kwargs["initial_prompt"], TRANSCRIBE_INITIAL_PROMPT)
        # Second final: previous final text chained into the prompt.
        second_kwargs = calls[1][1]
        self.assertIsNotNone(second_kwargs.get("initial_prompt"),
                             "second final must receive a chained prompt")
        self.assertIn("Hello world", second_kwargs["initial_prompt"])

    @patch("src.transcriber.WhisperModel")
    def test_partial_transcribe_chains_previous_final(self, mock_whisper):
        """Partial chunks after a final must also carry the previous final's
        text as context."""
        import threading
        import time
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        mock_seg = MagicMock()
        mock_seg.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        from src.transcriber import start_transcriber
        asr_q = queue.Queue()
        tr_q = queue.Queue()
        ui_q = queue.Queue()

        t = threading.Thread(target=start_transcriber, args=(asr_q, tr_q, ui_q))
        t.start()

        def wait_for_calls(n, timeout=5.0):
            deadline = time.time() + timeout
            while time.time() < deadline:
                if len(mock_model.transcribe.call_args_list) >= n:
                    return
                time.sleep(0.05)
            raise AssertionError(
                f"expected {n} transcribe calls, got {len(mock_model.transcribe.call_args_list)}"
            )

        asr_q.put((b"audio_data_1", 16000, True))
        wait_for_calls(1)
        asr_q.put((b"audio_data_2", 16000, False))
        wait_for_calls(2)
        asr_q.put("QUIT")
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "transcriber thread did not exit")

        calls = mock_model.transcribe.call_args_list
        self.assertGreaterEqual(len(calls), 2)
        second_kwargs = calls[1][1]
        self.assertIn("Hello world", second_kwargs["initial_prompt"])


if __name__ == "__main__":
    unittest.main()
