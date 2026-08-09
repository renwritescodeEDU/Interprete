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

    # Mock PyAudio to avoid needing real audio hardware in tests
    with patch("pyaudio.PyAudio") as mock_pyaudio_cls:
        mock_pyaudio = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio_cls.return_value = mock_pyaudio
        mock_pyaudio.open.return_value = mock_stream

        start_audio_capture(test_queue)

        item = test_queue.get(timeout=2)
        assert isinstance(item, tuple)
        assert len(item) == 2
        audio_array, sample_rate = item
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000

        mock_pyaudio.open.assert_called_once()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pyaudio.terminate.assert_called_once()

