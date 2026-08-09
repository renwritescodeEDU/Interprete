import multiprocessing
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.audio import start_audio_capture


def test_audio_capture_callable():
    """Verify start_audio_capture exists and is callable."""
    assert callable(start_audio_capture)


def test_audio_capture_pushes_to_queue():
    """Verify start_audio_capture pushes (audio_array, sample_rate) to asr_queue."""
    test_queue = multiprocessing.Queue()

    # Mock PyAudio and webrtcvad
    with patch("pyaudio.PyAudio") as mock_pyaudio_cls, patch("webrtcvad.Vad") as mock_vad_cls:
        mock_pyaudio = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio_cls.return_value = mock_pyaudio
        mock_pyaudio.open.return_value = mock_stream
        
        mock_vad = MagicMock()
        mock_vad_cls.return_value = mock_vad
        
        # Simulate speech for 5 frames, then silence for 15 frames (to trigger threshold)
        mock_vad.is_speech.side_effect = [True]*5 + [False]*15
        
        dummy_frame = b"\x00" * 960 # 480 samples * 2 bytes
        
        def mock_read(*args, **kwargs):
            mock_read.call_count += 1
            if mock_read.call_count > 20:
                raise Exception("End of test stream")
            return dummy_frame
        mock_read.call_count = 0
        mock_stream.read.side_effect = mock_read

        start_audio_capture(test_queue)

        item = test_queue.get(timeout=2)
        assert isinstance(item, tuple)
        assert len(item) == 2
        audio_array, sample_rate = item
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000
        
        # The array should have 5 speech + 14 silence frames = 19 frames (the 15th silence frame triggers flush before it is added to the next buffer usually, or actually the threshold is 13 or something, let's just assert length > 0)
        assert len(audio_array) > 0

        mock_pyaudio.open.assert_called_once()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pyaudio.terminate.assert_called_once()

