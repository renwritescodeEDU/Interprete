import multiprocessing
import queue
import pytest
from unittest.mock import MagicMock, patch
from src.translator import start_translator

def test_translator_callable():
    """Verify start_translator exists and is callable."""
    assert callable(start_translator)

@patch("src.translator.ollama.chat")
def test_translator_processes_queue(mock_ollama_chat):
    """Verify start_translator pulls from translation_queue and pushes translation event."""
    
    # Mock Ollama chat
    def side_effect_chat(model, messages, **kwargs):
        content = messages[-1]['content']
        try:
            text_to_translate = content.split("Text to translate:\n")[-1].strip()
        except IndexError:
            text_to_translate = content
            
        if "Hello" in text_to_translate:
            return {'message': {'content': '{"translation": "Hola_Ollama"}'}}
        elif "Hola" in text_to_translate:
            return {'message': {'content': '{"translation": "Hello_Ollama"}'}}
        elif "hi" in text_to_translate:
            return {'message': {'content': '{"translation": "hi"}'}} # Pre-warm
        return {'message': {'content': '{"translation": "Unknown"}'}}
        
    mock_ollama_chat.side_effect = side_effect_chat

    translation_queue = multiprocessing.Queue()
    ui_queue = multiprocessing.Queue()

    # Test English to Spanish
    translation_queue.put(("Hello", "en"))
    # Poison pill to exit loop
    translation_queue.put(None)

    start_translator(translation_queue, ui_queue)

    # First message is the "translator ready" status notification
    status_msg = ui_queue.get(timeout=2)
    assert isinstance(status_msg, dict)
    assert status_msg.get("type") == "status"
    assert status_msg.get("process") == "translator"
    assert status_msg.get("status") == "ready"

    # Second message is the translation result
    item1 = ui_queue.get(timeout=2)
    assert isinstance(item1, dict)
    assert item1["type"] == "translation"
    assert item1["original"] == "Hello"
    assert item1["translated"] == "Hola_Ollama"
    assert "latency" in item1
