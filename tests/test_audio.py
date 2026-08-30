import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch
import queue
import numpy as np
import time

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio import (
    list_audio_devices,
    _open_stream,
    _process_audio_frames,
    start_audio_capture,
    RATE,
    CHUNK,
)


class TestAudioModule(unittest.TestCase):

    def test_audio_capture_callable(self):
        """Verify that start_audio_capture is a callable function."""
        self.assertTrue(callable(start_audio_capture))

    @patch('src.audio.pyaudio.PyAudio')
    def test_list_audio_devices(self, mock_pyaudio):
        """Mock PyAudio, verify device classification (input/output/both), default detection."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        
        mock_p.get_default_input_device_info.return_value = {"index": 1}
        mock_p.get_device_count.return_value = 4
        
        def get_device_info(idx):
            if idx == 0:
                return {"name": "Dev0", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 44100}
            elif idx == 1:
                return {"name": "Dev1", "maxInputChannels": 1, "maxOutputChannels": 2, "defaultSampleRate": 16000}
            elif idx == 2:
                return {"name": "Dev2", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000}
            elif idx == 3:
                return {"name": "Dev3", "maxInputChannels": 0, "maxOutputChannels": 0, "defaultSampleRate": 44100}
                
        mock_p.get_device_info_by_index.side_effect = get_device_info
        
        devices = list_audio_devices()
        self.assertEqual(len(devices), 3)  # device 3 skipped due to no channels
        
        self.assertEqual(devices[0]["type"], "output")
        self.assertFalse(devices[0]["is_default"])
        
        self.assertEqual(devices[1]["type"], "both")
        self.assertTrue(devices[1]["is_default"])
        
        self.assertEqual(devices[2]["type"], "input")
        self.assertFalse(devices[2]["is_default"])

    @patch('src.audio.pyaudio.PyAudio')
    def test_audio_capture_pushes_to_queue(self, mock_pyaudio):
        """START+FINISH+QUIT, verify 4-element tuple with timing dict."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        mock_stream = MagicMock()
        mock_p.open.return_value = mock_stream
        
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        
        start_ts = time.time()
        stop_ts = start_ts + 2.0
        
        # Simulate START, wait for a bit, FINISH, then QUIT
        control_queue.put(("START", start_ts))
        control_queue.put(("FINISH", stop_ts))
        control_queue.put("QUIT")
        
        # Mock some read data so there are frames to process
        mock_stream.read.return_value = b'\x00' * 960
        
        start_audio_capture(asr_queue, control_queue)
        
        found_final = False
        while not asr_queue.empty():
            item = asr_queue.get()
            if len(item) == 4:
                found_final = True
                audio_array, rate, is_final, timing = item
                self.assertEqual(rate, RATE)
                self.assertTrue(is_final)
                self.assertIsInstance(timing, dict)
                self.assertEqual(timing["recording_start"], start_ts)
                self.assertEqual(timing["recording_stop"], stop_ts)
                self.assertIsInstance(audio_array, np.ndarray)
                
        self.assertTrue(found_final, "Did not find final 4-element tuple in asr_queue")

    @patch('src.audio.pyaudio.PyAudio')
    def test_audio_capture_with_device_index(self, mock_pyaudio):
        """Verify device_index passed to p.open()."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        mock_stream = MagicMock()
        mock_p.open.return_value = mock_stream
        
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        control_queue.put("QUIT")
        
        target_device_index = 5
        start_audio_capture(asr_queue, control_queue, device_index=target_device_index)
        
        mock_p.open.assert_called_once()
        open_kwargs = mock_p.open.call_args[1]
        self.assertEqual(open_kwargs.get("input_device_index"), target_device_index)

    @patch('src.audio.pyaudio.PyAudio')
    def test_audio_capture_waits_when_no_device(self, mock_pyaudio):
        """Verify no-device startup enters a 'waiting' state instead of dying."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p

        # Make p.open raise an exception both for target rate and fallback
        mock_p.open.side_effect = Exception("Mocked stream error")
        mock_p.get_default_input_device_info.return_value = {"defaultSampleRate": 44100}

        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        ui_queue = queue.Queue()
        control_queue.put("QUIT")

        start_audio_capture(asr_queue, control_queue, ui_queue=ui_queue)

        msg = ui_queue.get_nowait()
        self.assertEqual(msg["type"], "status")
        self.assertEqual(msg["process"], "audio")
        self.assertEqual(msg["status"], "waiting")
        # The worker must not have died — it returned only because of QUIT.
        self.assertTrue(ui_queue.empty())

    @patch('src.audio.pyaudio.PyAudio')
    def test_audio_capture_detects_device_later(self, mock_pyaudio):
        """Worker in 'waiting' state must transition to 'ready' when a device appears."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        mock_stream = MagicMock()
        # _open_stream tries twice per attempt: first open fails (no device),
        # the retry succeeds once a device is plugged in.
        mock_p.open.side_effect = [
            Exception("no device"), Exception("no device"),   # initial attempt fails
            Exception("no device"), mock_stream,              # retry succeeds
        ]
        mock_p.get_default_input_device_info.return_value = {"defaultSampleRate": 44100}

        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        ui_queue = queue.Queue()

        import threading
        worker = threading.Thread(
            target=start_audio_capture,
            args=(asr_queue, control_queue, ui_queue),
            daemon=True,
        )
        with patch('src.audio.DEVICE_RETRY_INTERVAL', 0.0):
            worker.start()
            time.sleep(0.3)  # let the initial failure and the retry both run
            control_queue.put("QUIT")
            worker.join(timeout=5)

        statuses = []
        while not ui_queue.empty():
            msg = ui_queue.get()
            if msg.get("type") == "status" and msg.get("process") == "audio":
                statuses.append(msg["status"])
        self.assertIn("waiting", statuses, f"expected a waiting status, got {statuses}")
        self.assertIn("ready", statuses, f"expected a ready status after retry, got {statuses}")

    @patch('src.audio.pyaudio.PyAudio')
    def test_audio_capture_reports_truncation(self, mock_pyaudio):
        """Verify a recording exceeding MAX_RECORDING_MINUTES emits a truncated event once."""
        from src.audio import MAX_RECORDING_MINUTES
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        mock_stream = MagicMock()
        mock_stream.read.return_value = b'\x00' * 960  # 480 samples @ int16
        mock_stream.get_read_available.side_effect = OSError("flush error")
        mock_p.open.return_value = mock_stream
        
        asr_queue = queue.Queue()
        ui_queue = queue.Queue()
        
        max_frames = int(MAX_RECORDING_MINUTES * 60 * 16000 / 480)
        seq = [("START", 0.0)] + [None] * (max_frames + 2000) + [("FINISH", 1.0), "QUIT"]
        
        class SequenceControl:
            def __init__(self, sequence):
                self._seq = list(sequence)
                self._i = 0
            def get_nowait(self):
                if self._i >= len(self._seq):
                    raise queue.Empty
                item = self._seq[self._i]
                self._i += 1
                if item is None:
                    raise queue.Empty
                return item
        
        start_audio_capture(asr_queue, SequenceControl(seq), ui_queue=ui_queue)
        
        truncated_events = []
        while not ui_queue.empty():
            msg = ui_queue.get()
            if msg.get("type") == "truncated":
                truncated_events.append(msg)
        
        self.assertEqual(len(truncated_events), 1, "truncation must be reported exactly once")
        self.assertEqual(truncated_events[0]["max_minutes"], MAX_RECORDING_MINUTES)
        self.assertGreaterEqual(truncated_events[0]["dropped_seconds"], 0.0)

    def test_process_audio_frames_no_resampling(self):
        """Direct call with actual_rate==RATE"""
        # 1 frame = 2 bytes (int16). Let's make 2 frames (4 bytes).
        frames = [b'\x00\x00', b'\x00\x40'] # 0 and 16384 in int16 LE
        actual_rate = RATE
        
        audio_array = _process_audio_frames(frames, actual_rate)
        
        self.assertEqual(len(audio_array), 2)
        self.assertAlmostEqual(audio_array[0], 0.0)
        self.assertAlmostEqual(audio_array[1], 16384 / 32768.0)

    @patch('src.audio.scipy.signal.resample_poly')
    def test_process_audio_frames_resampling(self, mock_resample):
        """Direct call with actual_rate!=RATE (e.g. 48000)."""
        mock_resample.return_value = np.array([0.5, -0.5], dtype=np.float32)
        
        frames = [b'\x00\x00' * 3] # 3 samples
        actual_rate = 48000
        
        audio_array = _process_audio_frames(frames, actual_rate)
        
        mock_resample.assert_called_once()
        args, kwargs = mock_resample.call_args
        self.assertEqual(args[1], 1) # numerator for 16000/48000 -> 1/3
        self.assertEqual(args[2], 3) # denominator
        
        # Audio array should be the mocked return value
        np.testing.assert_array_equal(audio_array, np.array([0.5, -0.5], dtype=np.float32))

    def test_open_stream_fallback(self):
        """Mock p.open to fail at 16kHz, succeed at native rate."""
        mock_p = MagicMock()
        mock_stream = MagicMock()
        
        def open_side_effect(**kwargs):
            if kwargs.get("rate") == RATE:
                raise ValueError("Format not supported")
            return mock_stream
            
        mock_p.open.side_effect = open_side_effect
        mock_p.get_default_input_device_info.return_value = {"defaultSampleRate": 48000}
        
        stream, actual_rate, buffer_size = _open_stream(mock_p)
        
        self.assertEqual(stream, mock_stream)
        self.assertEqual(actual_rate, 48000)
        self.assertEqual(buffer_size, int(CHUNK * 48000 / RATE))
        self.assertEqual(mock_p.open.call_count, 2)

    @patch('src.audio.pyaudio.PyAudio')
    def test_audio_capture_set_device(self, mock_pyaudio):
        """Send SET_DEVICE command, verify stream reopened."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        
        # Return unique stream mocks each time open is called
        mock_stream_1 = MagicMock()
        mock_stream_2 = MagicMock()
        mock_p.open.side_effect = [mock_stream_1, mock_stream_2]
        
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        
        control_queue.put(("SET_DEVICE", 3))
        control_queue.put("QUIT")
        
        start_audio_capture(asr_queue, control_queue)
        
        self.assertEqual(mock_p.open.call_count, 2)
        # Verify second open call used device index 3
        open_kwargs = mock_p.open.call_args[1]
        self.assertEqual(open_kwargs.get("input_device_index"), 3)
        mock_stream_1.stop_stream.assert_called_once()
        mock_stream_1.close.assert_called_once()

    @patch('src.audio.pyaudio.PyAudio')
    def test_audio_capture_timing_propagation(self, mock_pyaudio):
        """Verify timing dict has recording_start and recording_stop keys."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        mock_stream = MagicMock()
        mock_stream.get_read_available.return_value = 0
        mock_stream.read.return_value = b'\x00' * 960 # dummy bytes
        mock_p.open.return_value = mock_stream
        
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        
        control_queue.put("START")
        control_queue.put("FINISH")
        control_queue.put("QUIT")
        
        start_audio_capture(asr_queue, control_queue)
        
        item = None
        while not asr_queue.empty():
            q_item = asr_queue.get()
            if len(q_item) == 4:
                item = q_item
                
        self.assertIsNotNone(item)
        timing = item[3]
        self.assertIn("recording_start", timing)
        self.assertIn("recording_stop", timing)
        self.assertIsNotNone(timing["recording_start"])
        self.assertIsNotNone(timing["recording_stop"])
        self.assertTrue(timing["recording_stop"] >= timing["recording_start"])

    @patch('src.audio.pyaudio.PyAudio')
    def test_audio_capture_quit_command(self, mock_pyaudio):
        """Verify QUIT exits cleanly."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        mock_stream = MagicMock()
        mock_p.open.return_value = mock_stream
        
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        
        control_queue.put("QUIT")
        
        start_audio_capture(asr_queue, control_queue)
        
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_p.terminate.assert_called_once()

    @patch('src.audio.pyaudio.PyAudio')
    def test_list_audio_devices_no_default(self, mock_pyaudio):
        """Mock get_default_input_device_info to raise IOError."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        
        mock_p.get_default_input_device_info.side_effect = IOError("No default input")
        mock_p.get_device_count.return_value = 1
        
        mock_p.get_device_info_by_index.return_value = {
            "name": "Dev1", "maxInputChannels": 1, "maxOutputChannels": 0, "defaultSampleRate": 16000
        }
        
        devices = list_audio_devices()
        
        self.assertEqual(len(devices), 1)
        self.assertFalse(devices[0]["is_default"])
        self.assertEqual(devices[0]["type"], "input")


# --- Block A: Auto-commit by silence (Phase 8) ---

FRAME_SPEECH = (10000).to_bytes(2, 'little', signed=True) * 480
FRAME_SILENCE = b'\x00' * 960
# With _open_stream mock → actual_rate=16000, buffer_size=480.
# Each frame = 30 ms. 67 frames silence = 2.01 s. 100 total frames = 3.0 s.


class TestAutoCommit(unittest.TestCase):
    """Auto-commit by silence (Phase 8) — tests run in a thread because
    start_audio_capture blocks on the recording loop."""

    def _reader(self, *segments):
        """Returns a stream.read side_effect built from (speech, silence)
        frame-count segments, then silence forever (loop never starves)."""
        plan = []
        for speech, silence in segments:
            plan.extend([FRAME_SPEECH] * speech)
            plan.extend([FRAME_SILENCE] * silence)
        idx = [0]
        def read(*args, **kwargs):
            i = idx[0]
            idx[0] += 1
            if i < len(plan):
                return plan[i]
            return FRAME_SILENCE
        return read

    def _run_capture(self, reader, start_ts, control_queue, asr_queue):
        mock_p = MagicMock()
        mock_p.open.return_value = MagicMock()
        mock_p.open.return_value.get_read_available.return_value = 0
        mock_p.open.return_value.read.side_effect = reader
        with patch('src.audio.pyaudio.PyAudio', return_value=mock_p):
            start_audio_capture(asr_queue, control_queue)

    def _collect_finals(self, asr_queue, timeout=5.0, quiet=1.5):
        """Collect final 4-tuples, stopping early once no new final arrives
        within `quiet` seconds (the auto-commit sequence is complete)."""
        deadline = time.time() + timeout
        finals = []
        last_hit = time.time()
        while time.time() < deadline:
            try:
                item = asr_queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 4:
                    finals.append(item)
                    last_hit = time.time()
            except queue.Empty:
                if time.time() - last_hit > quiet:
                    break
                time.sleep(0.02)
        return finals

    def test_audio_auto_commit_on_long_silence(self):
        """50 speech frames (1.5s) + 300 silence (9s) → auto-commit at ~3.5s
        (segment=3.5s, silence=2.0s). Final 4-tuple must appear before FINISH."""
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        start_ts = time.time()
        control_queue.put(("START", start_ts))

        t = threading.Thread(target=self._run_capture, args=(
            self._reader((50, 300)), start_ts, control_queue, asr_queue))
        t.start()

        auto_final = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                item = asr_queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 4 and item[2] is True:
                    auto_final = item
                    break
            except queue.Empty:
                time.sleep(0.05)

        control_queue.put(("FINISH", time.time()))
        control_queue.put("QUIT")
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "capture thread did not exit")
        self.assertIsNotNone(auto_final, "auto-commit final never emitted")

        audio_array, rate, is_final, timing = auto_final
        self.assertTrue(is_final)
        self.assertEqual(timing["recording_start"], start_ts)
        self.assertGreaterEqual(timing["recording_stop"], start_ts)

        # Verify FINISH did not add a second final
        finals = self._collect_finals(asr_queue, timeout=2.0)
        self.assertEqual(len(finals), 0, "FINISH should not emit a second final after auto-commit with silence")

    def test_audio_no_auto_commit_without_speech(self):
        """All silence → never auto-commit. Only the FINISH final should appear."""
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        start_ts = time.time()
        control_queue.put(("START", start_ts))

        t = threading.Thread(target=self._run_capture, args=(
            self._reader((0, 999999)), start_ts, control_queue, asr_queue))
        t.start()

        time.sleep(0.5)  # enough silence (>2s) to trigger commit if speech were present
        control_queue.put(("FINISH", time.time()))
        control_queue.put("QUIT")
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "capture thread did not exit")

        finals = self._collect_finals(asr_queue, timeout=2.0)
        self.assertEqual(len(finals), 1, "expected exactly 1 final (from FINISH, not auto-commit)")
        _, _, is_final, _ = finals[0]
        self.assertTrue(is_final)

    def test_audio_auto_commit_resets_segment(self):
        """50 speech + 250 silence + 30 speech + 250 silence → 2 auto-commits.
        The second commit must have recording_start > the first commit's."""
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        start_ts = time.time()
        control_queue.put(("START", start_ts))

        t = threading.Thread(target=self._run_capture, args=(
            self._reader((50, 250), (30, 250)), start_ts, control_queue, asr_queue))
        t.start()

        all_finals = self._collect_finals(asr_queue, timeout=8.0)
        control_queue.put("QUIT")
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "capture thread did not exit")

        self.assertEqual(len(all_finals), 2,
                         f"expected 2 auto-commits, got {len(all_finals)}")
        t1 = all_finals[0][3]["recording_start"]
        t2 = all_finals[1][3]["recording_start"]
        self.assertGreater(t2, t1, "second segment must have a later recording_start")

    def test_audio_auto_commit_thought_pause_not_split(self):
        """30 speech + 60 silence (1.8s thought pause) + 30 speech + 400 silence.
        The pause must NOT split the utterance — only 1 auto-commit at the end."""
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        start_ts = time.time()
        control_queue.put(("START", start_ts))

        t = threading.Thread(target=self._run_capture, args=(
            self._reader((30, 60), (30, 400)), start_ts, control_queue, asr_queue))
        t.start()

        all_finals = self._collect_finals(asr_queue, timeout=8.0)
        control_queue.put("QUIT")
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "capture thread did not exit")

        self.assertEqual(len(all_finals), 1,
                         f"thought pause should not split — got {len(all_finals)} finals")

    def test_audio_finish_after_commit_adds_nothing(self):
        """50 speech + 300 silence → auto-commit. FINISH with residual silence
        must not emit a second (empty) final."""
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        start_ts = time.time()
        control_queue.put(("START", start_ts))

        t = threading.Thread(target=self._run_capture, args=(
            self._reader((50, 300)), start_ts, control_queue, asr_queue))
        t.start()

        auto_final = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                item = asr_queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 4 and item[2] is True:
                    auto_final = item
                    break
            except queue.Empty:
                time.sleep(0.05)
        self.assertIsNotNone(auto_final)

        time.sleep(0.3)  # let residual silence frames accumulate
        control_queue.put(("FINISH", time.time()))
        time.sleep(0.1)  # give time for processing
        control_queue.put("QUIT")
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())

        # Collect any finals that appeared AFTER the auto-commit
        post_finals = []
        while not asr_queue.empty():
            try:
                item = asr_queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 4:
                    post_finals.append(item)
            except queue.Empty:
                break
        self.assertEqual(len(post_finals), 0,
                         f"FINISH with residual silence should not add finals; got {len(post_finals)}")


if __name__ == '__main__':
    unittest.main()
