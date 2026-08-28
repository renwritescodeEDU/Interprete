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
        
        text, latency = translator.translate_ollama("Hello", "English", "Spanish", [])
        
        self.assertTrue(text.startswith("[LLM Error: Ollama is down]"))
        self.assertEqual(latency, 2.0)

    @patch('src.translator.translate_ollama')
    def test_process_translation_task(self, mock_translate_ollama):
        """Test task processing sends correct message to ui_queue."""
        mock_translate_ollama.return_value = ("Hola", 1.5)
        ui_queue = MagicMock()
        timing = {"audio_start": 0.5}
        
        # "Hello world" is clearly English → being translated to Spanish, guard won't trigger
        translator.process_translation_task(("Hello world", "en"), ["Context"], ui_queue, timing)
        
        self.assertIn("translation_start", timing)
        self.assertIn("translation_end", timing)
        
        ui_queue.put.assert_called_once()
        put_arg = ui_queue.put.call_args[0][0]
        self.assertEqual(put_arg["type"], "translation")
        self.assertEqual(put_arg["original"], "Hello world")
        self.assertEqual(put_arg["translated"], "Hola")
        self.assertEqual(put_arg["latency"], 1.5)
        self.assertEqual(put_arg["timing"], timing)

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.show')
    @patch('src.translator.ollama.chat')
    def test_translator_processes_queue(self, mock_chat, mock_show, mock_get_glossary):
        """Test full main loop processes translation."""
        mock_glossary = MagicMock()
        mock_get_glossary.return_value = mock_glossary
        mock_show.return_value = {}
        
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

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.concurrent.futures.ThreadPoolExecutor')
    @patch('src.translator.ollama.chat')
    def test_translator_context_history_limit(self, mock_chat, mock_executor_class, mock_get_glossary):
        """Test context history does not exceed 10 items."""
        mock_glossary = MagicMock()
        mock_get_glossary.return_value = mock_glossary
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
        # args = (process_translation_task, task, context_history, ui_queue, timing, glossary_mgr)
        context_history = last_call_args[2]
        self.assertEqual(len(context_history), 10)
        self.assertEqual(context_history[-1], "Msg 14")
        self.assertEqual(context_history[0], "Msg 5")

    @patch('src.translator.translate_ollama')
    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.chat')
    def test_translator_timing_propagation(self, mock_chat, mock_get_glossary, mock_translate_ollama):
        """Test timing dict receives start/end stamps."""
        mock_glossary = MagicMock()
        mock_get_glossary.return_value = mock_glossary
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
    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.chat')
    def test_translator_legacy_2element_tuple(self, mock_chat, mock_get_glossary, mock_translate_ollama):
        """Test 2-element tuple without timing works correctly."""
        mock_glossary = MagicMock()
        mock_get_glossary.return_value = mock_glossary
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
            speaker_note="Agent speaking.",
            glossary_section="Key terminology:\n- checking account = cuenta corriente",
            context_str="- Context1",
            text="Hello"
        )
        self.assertIn("English", formatted)
        self.assertIn("- Context1", formatted)
        self.assertIn("Hello", formatted)
        self.assertIn("checking account = cuenta corriente", formatted)

    def test_prompt_contains_formal_register_rules(self):
        """Test that the prompt enforces formal register (usted)."""
        self.assertIn("usted", translator.TRANSLATION_PROMPT_TEMPLATE)
        self.assertIn("REGISTER", translator.TRANSLATION_PROMPT_TEMPLATE)

    def test_prompt_contains_acronym_rules(self):
        """Test that the prompt has acronym expansion instructions."""
        self.assertIn("acronym", translator.TRANSLATION_PROMPT_TEMPLATE.lower())
        self.assertIn("APR", translator.TRANSLATION_PROMPT_TEMPLATE)

    def test_prompt_contains_compound_term_rules(self):
        """Test that the prompt has compound term translation rules."""
        self.assertIn("mother-in-law", translator.TRANSLATION_PROMPT_TEMPLATE)
        self.assertIn("suegra", translator.TRANSLATION_PROMPT_TEMPLATE)
        self.assertIn("checking account", translator.TRANSLATION_PROMPT_TEMPLATE)
        self.assertIn("cuenta corriente", translator.TRANSLATION_PROMPT_TEMPLATE)

    def test_model_upgraded(self):
        """Test that the model has been upgraded from 1.5b."""
        self.assertNotEqual(translator.LLM_MODEL, "qwen2.5:1.5b")
        self.assertEqual(translator.LLM_MODEL, "qwen2.5:3b")

    def test_context_limit_increased(self):
        """Test that context limit has been increased from 2."""
        self.assertGreaterEqual(translator.CONTEXT_LIMIT, 5)

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.show')
    @patch('src.translator.ollama.chat')
    def test_translator_prewarm_failure(self, mock_chat, mock_show, mock_get_glossary):
        """Test translator starts despite prewarm failure."""
        mock_glossary = MagicMock()
        mock_get_glossary.return_value = mock_glossary
        mock_show.return_value = {}
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

    @patch('src.translator.ollama.chat')
    def test_translate_with_glossary_manager(self, mock_chat):
        """Test that glossary manager is properly integrated into translation."""
        mock_chat.return_value = {'message': {'content': '{"translation": "cuenta corriente"}'}}
        self.mock_time.side_effect = [1.0, 2.0]

        mock_glossary = MagicMock()
        mock_glossary.build_glossary_prompt_section.return_value = (
            "[Industry: Finanzas y Banca]\n\n"
            "Key terminology:\n- checking account = cuenta corriente"
        )
        
        text, latency = translator.translate_ollama(
            "checking account", "English", "Spanish", 
            ["I want to open an account"],
            glossary_manager=mock_glossary
        )
        
        self.assertEqual(text, "cuenta corriente")
        mock_glossary.build_glossary_prompt_section.assert_called_once()
        # Verify the glossary section was included in the prompt
        call_args = mock_chat.call_args
        prompt_content = call_args[1]['messages'][0]['content']
        self.assertIn("checking account = cuenta corriente", prompt_content)


class TestGlossaryModule(unittest.TestCase):
    """Test the glossary module independently."""

    def test_glossary_module_importable(self):
        """Test that the glossary module can be imported."""
        from src.glossary import GlossaryManager, get_glossary_manager
        self.assertTrue(callable(GlossaryManager))
        self.assertTrue(callable(get_glossary_manager))

    def test_glossary_manager_initialization(self):
        """Test GlossaryManager initializes cleanly."""
        from src.glossary import GlossaryManager
        mgr = GlossaryManager("/nonexistent/path")
        mgr.load_all()
        # Should not crash, just warn
        self.assertTrue(mgr._loaded)

    def test_glossary_manager_loads_real_glossaries(self):
        """Test that GlossaryManager loads the actual glossary files."""
        from src.glossary import GlossaryManager, GLOSSARIES_DIR
        if not os.path.isdir(GLOSSARIES_DIR):
            self.skipTest("Glossaries directory not found")
        
        mgr = GlossaryManager()
        mgr.load_all()
        
        # Should have loaded at least some glossaries
        self.assertTrue(len(mgr._glossaries) > 0 or len(mgr._acronyms_master) > 0,
                        "No glossaries loaded from the glossaries directory")

    def test_industry_detection(self):
        """Test that industry detection works for finance keywords."""
        from src.glossary import GlossaryManager
        mgr = GlossaryManager()
        mgr.load_all()
        
        if not mgr._glossaries:
            self.skipTest("No glossaries loaded")
        
        # Test with banking context
        result = mgr.detect_industry([
            "Would you like to open a checking account?",
            "I need to check my balance and make a deposit"
        ])
        # Should detect finance/banking
        if result:
            self.assertIn("finance", result.lower().replace("_", " ") + " " + 
                          mgr._glossaries.get(result, {}).get("display_name", "").lower())

    def test_get_relevant_terms(self):
        """Test term extraction from input text."""
        from src.glossary import GlossaryManager
        mgr = GlossaryManager()
        mgr._common_terms = {
            "mother-in-law": "suegra",
            "husband": "esposo",
            "date of birth": "fecha de nacimiento"
        }
        mgr._loaded = True
        
        result = mgr.get_relevant_terms("I live with my mother-in-law and husband")
        self.assertIn("suegra", result)
        self.assertIn("esposo", result)
        self.assertNotIn("fecha de nacimiento", result)

    def test_get_relevant_acronyms(self):
        """Test acronym extraction from input text."""
        from src.glossary import GlossaryManager
        mgr = GlossaryManager()
        mgr._acronyms_master = {
            "CD": {"full_en": "Certificate of Deposit", "es": "CD (Certificado de Depósito)"},
            "APR": {"full_en": "Annual Percentage Rate", "es": "TAP (Tasa Anual de Porcentaje)"},
            "DNA": {"full_en": "Deoxyribonucleic Acid", "es": "ADN (Ácido Desoxirribonucleico)"}
        }
        mgr._loaded = True
        
        result = mgr.get_relevant_acronyms("What is the APR on this CD account?")
        self.assertIn("APR", result)
        self.assertIn("CD", result)
        self.assertNotIn("DNA", result)


if __name__ == '__main__':
    unittest.main()
