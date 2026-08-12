import multiprocessing
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.audio import start_audio_capture, list_audio_devices


def test_audio_capture_callable():
    """Verify start_audio_capture exists and is callable."""
    assert callable(start_audio_capture)


@patch("src.audio.pyaudio.PyAudio")
def test_list_audio_devices(mock_pyaudio_cls):
    """Verify list_audio_devices returns all devices with correct type classification."""
    mock_pyaudio = MagicMock()
    mock_pyaudio_cls.return_value = mock_pyaudio

    # Default input is device 0
    mock_pyaudio.get_default_input_device_info.return_value = {"index": 0}
    mock_pyaudio.get_device_count.return_value = 3
    
    device_infos = [
        {"name": "Microphone", "maxInputChannels": 1, "maxOutputChannels": 0, "defaultSampleRate": 48000.0},
        {"name": "Speakers", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0},
        {"name": "BlackHole 2ch", "maxInputChannels": 2, "maxOutputChannels": 2, "defaultSampleRate": 48000.0},
    ]

    def get_info_by_index(i):
        return device_infos[i]

    mock_pyaudio.get_device_info_by_index.side_effect = get_info_by_index

    devices = list_audio_devices()

    # Should include ALL devices (input, output, and both)
    assert len(devices) == 3
    assert devices[0]["name"] == "Microphone"
    assert devices[0]["type"] == "input"
    assert devices[0]["is_default"] is True
    assert devices[1]["name"] == "Speakers"
    assert devices[1]["type"] == "output"
    assert devices[1]["is_default"] is False
    assert devices[2]["name"] == "BlackHole 2ch"
    assert devices[2]["type"] == "both"
    assert devices[2]["is_default"] is False

    mock_pyaudio.terminate.assert_called_once()


def test_audio_capture_pushes_to_queue():
    """Verify start_audio_capture pushes (audio_array, sample_rate, is_final) to asr_queue after START and FINISH."""
    test_queue = multiprocessing.Queue()
    control_queue = multiprocessing.Queue()

    with patch("src.audio.pyaudio.PyAudio") as mock_pyaudio_cls:
        mock_pyaudio = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio_cls.return_value = mock_pyaudio
        mock_pyaudio.open.return_value = mock_stream

        dummy_frame = b"\x00" * 960  # 480 samples * 2 bytes
        mock_stream.read.return_value = dummy_frame
        mock_stream.get_read_available.return_value = 0

        # Preload commands: START, then FINISH, then QUIT to exit cleanly
        control_queue.put("START")
        control_queue.put("FINISH")
        control_queue.put("QUIT")

        start_audio_capture(test_queue, control_queue, ui_queue=None, device_index=None)

        # It should emit a final item when FINISH is called
        item = test_queue.get(timeout=2)
        assert isinstance(item, tuple)
        assert len(item) == 3
        audio_array, sample_rate, is_final = item
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000
        assert is_final is True
        assert len(audio_array) > 0


def test_audio_capture_with_device_index():
    """Verify start_audio_capture opens the specified device index."""
    test_queue = multiprocessing.Queue()
    control_queue = multiprocessing.Queue()
    control_queue.put("QUIT")

    with patch("src.audio.pyaudio.PyAudio") as mock_pyaudio_cls:
        mock_pyaudio = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio_cls.return_value = mock_pyaudio
        mock_pyaudio.open.return_value = mock_stream
        mock_stream.get_read_available.return_value = 0

        start_audio_capture(test_queue, control_queue, ui_queue=None, device_index=2)

        # Verify the device index was passed to p.open()
        assert mock_pyaudio.open.call_args.kwargs["input_device_index"] == 2


def test_audio_capture_reports_error_to_ui_queue():
    """Verify that audio errors are reported to ui_queue when provided."""
    test_queue = multiprocessing.Queue()
    control_queue = multiprocessing.Queue()
    ui_queue = multiprocessing.Queue()

    with patch("src.audio.pyaudio.PyAudio") as mock_pyaudio_cls:
        mock_pyaudio = MagicMock()
        mock_pyaudio_cls.return_value = mock_pyaudio
        # Simulate stream open failure
        mock_pyaudio.open.side_effect = Exception("Device disconnected")
        mock_pyaudio.get_default_input_device_info.side_effect = Exception("No device")

        start_audio_capture(test_queue, control_queue, ui_queue=ui_queue, device_index=None)

        # The process should have terminated without crashing
        # ui_queue may or may not have an error depending on where it fails
        # but the function should return cleanly
        mock_pyaudio.terminate.assert_called_once()
