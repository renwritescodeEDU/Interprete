import multiprocessing
import queue
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.transcriber import start_transcriber


def test_transcriber_callable():
    """Verify start_transcriber exists and is callable."""
    assert callable(start_transcriber)


@patch("src.transcriber.WhisperModel")
def test_transcriber_processes_queue(mock_whisper_model_cls):
    """Verify start_transcriber pulls from asr_queue and pushes (text, lang) to translation_queue."""
    mock_model = MagicMock()
    mock_whisper_model_cls.return_value = mock_model
    
    # Mock transcription result: segments and info
    mock_segment = MagicMock()
    mock_segment.text = "Hello world"
    mock_info = MagicMock()
    mock_info.language = "en"
    
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    asr_queue = multiprocessing.Queue()
    translation_queue = multiprocessing.Queue()
    ui_queue = multiprocessing.Queue()

    dummy_audio = np.zeros(1024, dtype=np.float32)
    sample_rate = 16000
    asr_queue.put((dummy_audio, sample_rate, True))
    asr_queue.put(None) # Poison pill to terminate loop

    start_transcriber(asr_queue, translation_queue, ui_queue)

    item = translation_queue.get(timeout=2)
    assert isinstance(item, tuple)
    assert len(item) == 2
    text, lang = item
    assert text == "Hello world"
    assert lang == "en"
    
    mock_model.transcribe.assert_called_once()
