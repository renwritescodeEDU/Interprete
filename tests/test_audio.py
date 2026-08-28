import os
import sys
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
    def test_audio_capture_reports_error_to_ui_queue(self, mock_pyaudio):
        """Verify stream open failure is reported to ui_queue (no silent death)."""
        mock_p = MagicMock()
        mock_pyaudio.return_value = mock_p
        
        # Make p.open raise an exception both for target rate and fallback
        mock_p.open.side_effect = Exception("Mocked stream error")
        mock_p.get_default_input_device_info.return_value = {"defaultSampleRate": 44100}
        
        asr_queue = queue.Queue()
        control_queue = queue.Queue()
        ui_queue = queue.Queue()
        
        start_audio_capture(asr_queue, control_queue, ui_queue=ui_queue)
        
        msg = ui_queue.get_nowait()
        self.assertEqual(msg["type"], "error")
        self.assertIn("Mocked stream error", msg["message"])

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

if __name__ == '__main__':
    unittest.main()
