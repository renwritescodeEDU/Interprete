import multiprocessing
import queue
import numpy as np
import pytest

from src.transcriber import start_transcriber


def test_transcriber_callable():
    """Verify start_transcriber exists and is callable."""
    assert callable(start_transcriber)


def test_transcriber_processes_queue():
    """Verify start_transcriber pulls from asr_queue and pushes (text, lang) to translation_queue."""
    asr_queue = multiprocessing.Queue()
    translation_queue = multiprocessing.Queue()

    dummy_audio = np.zeros(1024, dtype=np.float32)
    sample_rate = 16000
    asr_queue.put((dummy_audio, sample_rate))

    start_transcriber(asr_queue, translation_queue)

    item = translation_queue.get(timeout=2)
    assert isinstance(item, tuple)
    assert len(item) == 2
    text, lang = item
    assert isinstance(text, str)
    assert isinstance(lang, str)
