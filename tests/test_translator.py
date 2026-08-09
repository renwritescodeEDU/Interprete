import multiprocessing
import queue
import pytest
from unittest.mock import MagicMock, patch

from src.translator import start_translator


def test_translator_callable():
    """Verify start_translator exists and is callable."""
    assert callable(start_translator)


@patch("src.translator.pipeline")
@patch("src.translator.torch.backends.mps.is_available")
def test_translator_processes_queue(mock_mps_is_available, mock_pipeline):
    """Verify start_translator pulls from translation_queue and pushes (original, translated) to ui_queue."""
    mock_mps_is_available.return_value = False
    
    # Mock pipelines returning predetermined results
    mock_en_es = MagicMock()
    mock_en_es.return_value = [{'translation_text': 'Hola'}]
    
    mock_es_en = MagicMock()
    mock_es_en.return_value = [{'translation_text': 'Hello'}]
    
    def side_effect_pipeline(task, model, device, **kwargs):
        if "en-es" in model:
            return mock_en_es
        return mock_es_en
        
    mock_pipeline.side_effect = side_effect_pipeline

    translation_queue = multiprocessing.Queue()
    ui_queue = multiprocessing.Queue()

    # Test English to Spanish
    translation_queue.put(("Hello", "en"))
    # Test Spanish to English
    translation_queue.put(("Hola", "es"))
    # Poison pill to exit loop
    translation_queue.put(None)

    start_translator(translation_queue, ui_queue)

    item1 = ui_queue.get(timeout=2)
    assert isinstance(item1, dict)
    assert item1["original"] == "Hello"
    assert item1["translated"] == "Hola"

    item2 = ui_queue.get(timeout=2)
    assert isinstance(item2, dict)
    assert item2["original"] == "Hola"
    assert item2["translated"] == "Hello"
    
    mock_en_es.assert_called_once_with("Hello")
    mock_es_en.assert_called_once_with("Hola")
