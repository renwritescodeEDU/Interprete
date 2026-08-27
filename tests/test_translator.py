import sys
import os
import unittest
from unittest.mock import patch, MagicMock, ANY
import queue
import multiprocessing
import time
import json
import concurrent.futures

# Ensure the parent directory is in sys.path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import translator

class TestTranslator(unittest.TestCase):

    def setUp(self):
        self.patcher_time = patch('src.translator.time.time')
        self.mock_time = self.patcher_time.start()
        # Provide enough dummy time values for the tests
        self.mock_time.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    def tearDown(self):
        self.patcher_time.stop()

    def test_translator_callable(self):
        """Test that the core functions are callable."""
        self.assertTrue(callable(translator.translate_ollama))
        self.assertTrue(callable(translator.process_translation_task))
        self.assertTrue(callable(translator.start_translator))

    @patch('src.translator.ollama.chat')
    def test_translate_ollama_success(self, mock_chat):
        """Test successful translation returns text and latency."""
        mock_chat.return_value = {'message': {'content': '{"translation": "Hola"}'}}
        self.mock_time.side_effect = [1.0, 2.5]
        
        text, latency = translator.translate_ollama("Hello", "English", "Spanish", ["Prev Context"])
        
        self.assertEqual(text, "Hola")
        self.assertEqual(latency, 1.5)
        mock_chat.assert_called_once()
        args, kwargs = mock_chat.call_args
        self.assertEqual(kwargs['model'], translator.LLM_MODEL)
        self.assertEqual(kwargs['format'], 'json')
        self.assertIn("Prev Context", kwargs['messages'][0]['content'])

    @patch('src.translator.ollama.chat')
    def test_translate_ollama_error(self, mock_chat):
        """Test translation failure returns error string."""
        mock_chat.side_effect = Exception("Ollama is down")
        # setUp provides side_effect [1.0, 2.0, 3.0, ...]
        # start_t = 1.0, logger calls time() -> 2.0, end_t = 3.0 -> 3.0 - 1.0 = 2.0
        
        text, latency = translator.translate_ollama("Hello", "English", "Spanish", [])
        
        self.assertTrue(text.startswith("[LLM Error: Ollama is down]"))
        self.assertEqual(latency, 2.0)

    @patch('src.translator.translate_ollama')
    def test_process_translation_task(self, mock_translate_ollama):
        """Test task processing sends correct message to ui_queue."""
        mock_translate_ollama.return_value = ("Hola", 1.5)
        ui_queue = MagicMock()
        timing = {"audio_start": 0.5}
        
        self.mock_time.side_effect = [1.0, 2.5]
        translator.process_translation_task(("Hello", "en"), ["Context"], ui_queue, timing)
        
        self.assertEqual(timing["translation_start"], 1.0)
        self.assertEqual(timing["translation_end"], 2.5)
        
        ui_queue.put.assert_called_once()
        put_arg = ui_queue.put.call_args[0][0]
        self.assertEqual(put_arg["type"], "translation")
        self.assertEqual(put_arg["original"], "Hello")
        self.assertEqual(put_arg["translated"], "Hola")
        self.assertEqual(put_arg["latency"], 1.5)
        self.assertEqual(put_arg["timing"], timing)

    @patch('src.translator.ollama.chat')
    def test_translator_processes_queue(self, mock_chat):
        """Test full main loop processes translation."""
        # First call is warmup, second is actual translation
        mock_chat.side_effect = [
            {'message': {'content': '{"test":"hi"}'}},
            {'message': {'content': '{"translation": "Hola"}'}}
        ]
        
        tq = multiprocessing.Queue()
        uq = multiprocessing.Queue()
        
        tq.put(("Hello", "en", {}))
        tq.put(None) # Poison pill
        
        # This will block until the queue is processed and the executor shuts down
        translator.start_translator(tq, uq)
        
        # uq should have status ready, then translation
        msg1 = uq.get(timeout=1)
        self.assertEqual(msg1["type"], "status")
        self.assertEqual(msg1["status"], "ready")
        
        msg2 = uq.get(timeout=1)
        self.assertEqual(msg2["type"], "translation")
        self.assertEqual(msg2["original"], "Hello")
        self.assertEqual(msg2["translated"], "Hola")

    @patch('src.translator.concurrent.futures.ThreadPoolExecutor')
    @patch('src.translator.ollama.chat')
    def test_translator_context_history_limit(self, mock_chat, mock_executor_class):
        """Test context history does not exceed 10 items."""
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        tq = multiprocessing.Queue()
        uq = MagicMock()
        
        for i in range(15):
            tq.put((f"Msg {i}", "en", {}))
        tq.put(None)
        
        translator.start_translator(tq, uq)
        
        self.assertEqual(mock_executor.submit.call_count, 15)
        # Check the context history argument of the last call
        last_call_args = mock_executor.submit.call_args[0]
        # args = (process_translation_task, task, context_history, ui_queue, timing)
        context_history = last_call_args[2]
        self.assertEqual(len(context_history), 10)
        self.assertEqual(context_history[-1], "Msg 14")
        self.assertEqual(context_history[0], "Msg 5")

    @patch('src.translator.translate_ollama')
    @patch('src.translator.ollama.chat')
    def test_translator_timing_propagation(self, mock_chat, mock_translate_ollama):
        """Test timing dict receives start/end stamps."""
        mock_translate_ollama.return_value = ("Hola", 0.5)
        
        tq = multiprocessing.Queue()
        uq = multiprocessing.Queue()
        
        timing = {"start": 0.0}
        tq.put(("Hello", "en", timing))
        tq.put(None)
        
        translator.start_translator(tq, uq)
        
        msg1 = uq.get(timeout=1) # status
        msg2 = uq.get(timeout=1) # translation
        
        self.assertIn("translation_start", msg2["timing"])
        self.assertIn("translation_end", msg2["timing"])

    @patch('src.translator.translate_ollama')
    @patch('src.translator.ollama.chat')
    def test_translator_legacy_2element_tuple(self, mock_chat, mock_translate_ollama):
        """Test 2-element tuple without timing works correctly."""
        mock_translate_ollama.return_value = ("Hola", 0.5)
        
        tq = multiprocessing.Queue()
        uq = multiprocessing.Queue()
        
        tq.put(("Hello", "en"))
        tq.put(None)
        
        translator.start_translator(tq, uq)
        
        msg1 = uq.get(timeout=1) # status
        msg2 = uq.get(timeout=1) # translation
        
        self.assertIn("translation_start", msg2["timing"])
        self.assertEqual(msg2["original"], "Hello")

    def test_translation_prompt_template_format(self):
        """Test TRANSLATION_PROMPT_TEMPLATE string format."""
        formatted = translator.TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang="English",
            target_lang="Spanish",
            context_str="- Context1",
            text="Hello"
        )
        self.assertIn("English text to Spanish", formatted)
        self.assertIn("- Context1", formatted)
        self.assertIn("Text to translate:\nHello", formatted)

    @patch('src.translator.ollama.chat')
    def test_translator_prewarm_failure(self, mock_chat):
        """Test translator starts despite prewarm failure."""
        # First call is prewarm, raise error.
        mock_chat.side_effect = Exception("Ollama connection failed")
        
        tq = multiprocessing.Queue()
        uq = multiprocessing.Queue()
        
        tq.put(None) # Immediate exit
        
        translator.start_translator(tq, uq)
        
        # uq should have status ready
        msg = uq.get(timeout=1)
        self.assertEqual(msg["type"], "status")
        self.assertEqual(msg["status"], "ready")

if __name__ == '__main__':
    unittest.main()
