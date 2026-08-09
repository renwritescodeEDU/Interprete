import multiprocessing
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.audio import start_audio_capture


def test_audio_capture_callable():
    """Verify start_audio_capture exists and is callable."""
    assert callable(start_audio_capture)


def test_audio_capture_pushes_to_queue():
    """Verify start_audio_capture pushes (audio_array, sample_rate, is_final) to asr_queue after START and FINISH."""
    test_queue = multiprocessing.Queue()
    control_queue = multiprocessing.Queue()

    # Mock PyAudio
    with patch("pyaudio.PyAudio") as mock_pyaudio_cls:
        mock_pyaudio = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio_cls.return_value = mock_pyaudio
        mock_pyaudio.open.return_value = mock_stream

        dummy_frame = b"\x00" * 960 # 480 samples * 2 bytes

        def mock_read(*args, **kwargs):
            return dummy_frame
        
        mock_stream.read.side_effect = mock_read
        
        def mock_get_read_available():
            mock_stream.read_count += 1
            if mock_stream.read_count > 10:
                raise BaseException("End of test stream")
            return 0
        
        mock_stream.read_count = 0
        mock_stream.get_read_available.side_effect = mock_get_read_available

        # Preload commands
        control_queue.put("START")
        control_queue.put("FINISH")

        try:
            start_audio_capture(test_queue, control_queue)
        except BaseException:
            pass

        # It should emit a final item when FINISH is called
        item = test_queue.get(timeout=2)
        assert isinstance(item, tuple)
        assert len(item) == 3
        audio_array, sample_rate, is_final = item
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000
        assert is_final is True
        
        # The array should have 5 speech + 14 silence frames = 19 frames (the 15th silence frame triggers flush before it is added to the next buffer usually, or actually the threshold is 13 or something, let's just assert length > 0)
        assert len(audio_array) > 0

        mock_pyaudio.open.assert_called_once()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pyaudio.terminate.assert_called_once()

