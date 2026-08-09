import multiprocessing
import queue
import pytest

from src.translator import start_translator


def test_translator_callable():
    """Verify start_translator exists and is callable."""
    assert callable(start_translator)


def test_translator_processes_queue():
    """Verify start_translator pulls from translation_queue and pushes (original, translated) to ui_queue."""
    translation_queue = multiprocessing.Queue()
    ui_queue = multiprocessing.Queue()

    translation_queue.put(("Hello", "en"))

    start_translator(translation_queue, ui_queue)

    item = ui_queue.get(timeout=2)
    assert isinstance(item, tuple)
    assert len(item) == 2
    original_text, translated_text = item
    assert original_text == "Hello"
    assert isinstance(translated_text, str)
