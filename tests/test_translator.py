import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import multiprocessing

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
        """Test translation failure returns None (never a fake translation)."""
        mock_chat.side_effect = Exception("Ollama is down")
        
        text, latency = translator.translate_ollama("Hello", "English", "Spanish", [])
        
        self.assertIsNone(text)
        self.assertEqual(latency, 2.0)

    @patch('src.translator.ollama.chat')
    def test_translate_ollama_missing_translation_key(self, mock_chat):
        """Valid JSON without a 'translation' key must be treated as a failure."""
        mock_chat.return_value = {'message': {'content': '{"result": "Hola"}'}}
        # Extra values cover the time.time() consumed by logging's record timestamps.
        self.mock_time.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0]

        text, _ = translator.translate_ollama("Hello", "English", "Spanish", [])

        self.assertIsNone(text)

    @patch('src.translator.ollama.chat')
    def test_translate_ollama_rejects_implausible_length(self, mock_chat):
        """Output far outside the plausible length band must be rejected."""
        mock_chat.return_value = {'message': {'content': '{"translation": "%s"}' % ('x' * 200)}}
        self.mock_time.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0]

        text, _ = translator.translate_ollama("Hello", "English", "Spanish", [])

        self.assertIsNone(text)

    def test_validate_translation_bounds(self):
        """Direct unit tests for the output sanity filter."""
        self.assertTrue(translator._validate_translation("Hello world", "Hola mundo"))
        self.assertTrue(translator._validate_translation("OK", "De acuerdo"))
        self.assertTrue(translator._validate_translation("", "Whatever"))
        self.assertFalse(translator._validate_translation("Hello world", ""))
        self.assertFalse(translator._validate_translation("Hello world", "   "))
        self.assertFalse(translator._validate_translation("Hello world", "x" * 500))
        self.assertFalse(translator._validate_translation("Hello world", "H"))
        self.assertFalse(translator._validate_translation("Hello world", None))

    def test_detect_same_language_word_boundaries(self):
        """English markers must not fire inside Spanish words (e.g. 'necesito ' vs 'to ')."""
        # Regression: "necesito " contains "to " as a substring but not as a word
        self.assertFalse(translator._detect_same_language(
            "Buenos días, necesito ayuda con mi cuenta.", "English"))
        self.assertTrue(translator._detect_same_language(
            "el banco y la cuenta", "Spanish"))
        # Genuine English detection still works
        self.assertTrue(translator._detect_same_language(
            "I need to open an account", "English"))
        self.assertFalse(translator._detect_same_language(
            "I need to open an account", "Spanish"))

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

    @patch('src.translator.translate_ollama')
    def test_process_translation_task_same_language_emits_skipped(self, mock_translate_ollama):
        """Same-language text must emit a 'skipped' terminal event, never a silent return."""
        ui_queue = MagicMock()
        timing = {"audio_start": 0.5}

        translator.process_translation_task(
            ("el banco y la cuenta", "en"), ["Context"], ui_queue, timing
        )

        mock_translate_ollama.assert_not_called()
        ui_queue.put.assert_called_once()
        msg = ui_queue.put.call_args[0][0]
        self.assertEqual(msg["type"], "skipped")
        self.assertEqual(msg["reason"], "same_language")
        self.assertEqual(msg["original"], "el banco y la cuenta")

    @patch('src.translator.translate_ollama', return_value=(None, 1.0))
    def test_process_translation_task_failure_emits_error(self, mock_translate_ollama):
        """LLM failure must emit an 'error' event, never a fake translation."""
        ui_queue = MagicMock()
        translator.process_translation_task(
            ("Hello world", "en"), ["Context"], ui_queue, {"audio_start": 0.5}
        )
        msg = ui_queue.put.call_args[0][0]
        self.assertEqual(msg["type"], "error")

    @patch('src.translator.translate_ollama', side_effect=RuntimeError("boom"))
    def test_process_translation_task_exception_emits_error(self, mock_translate_ollama):
        """Unexpected exceptions in the worker must surface as an error event."""
        ui_queue = MagicMock()
        translator.process_translation_task(
            ("Hello world", "en"), ["Context"], ui_queue, {"audio_start": 0.5}
        )
        msg = ui_queue.put.call_args[0][0]
        self.assertEqual(msg["type"], "error")
        self.assertIn("boom", msg["message"])

    @patch('src.translator.translate_ollama', return_value=("el banco y la cuenta", 1.5))
    def test_process_translation_task_rejects_source_language_echo(self, mock_translate_ollama):
        """An output that is still in the source language must be rejected, not displayed."""
        ui_queue = MagicMock()
        translator.process_translation_task(
            ("el banco y la cuenta", "es"), ["Context"], ui_queue, {"audio_start": 0.5}
        )
        # Translated target is English, but the model echoed the Spanish input
        msg = ui_queue.put.call_args[0][0]
        self.assertEqual(msg["type"], "error")
        self.assertIn("English", msg["message"])

    @patch('src.translator.translate_ollama', return_value=("the bank and the account", 1.5))
    def test_process_translation_task_accepts_target_language_output(self, mock_translate_ollama):
        """A genuine English output must be delivered as a translation event."""
        ui_queue = MagicMock()
        translator.process_translation_task(
            ("el banco y la cuenta", "es"), ["Context"], ui_queue, {"audio_start": 0.5}
        )
        msg = ui_queue.put.call_args[0][0]
        self.assertEqual(msg["type"], "translation")
        self.assertEqual(msg["translated"], "the bank and the account")

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
        """Context history is capped at 10 bilingual pairs and never contains the current text."""
        mock_glossary = MagicMock()
        mock_get_glossary.return_value = mock_glossary
        mock_chat.return_value = {'message': {'content': '{"translation": "Hola"}'}}
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Execute tasks inline so shared context actually accumulates
        def fake_submit(fn, *args, **kwargs):
            fn(*args, **kwargs)
            return MagicMock()
        mock_executor.submit.side_effect = fake_submit

        self.mock_time.side_effect = lambda: 1.0

        tq = multiprocessing.Queue()
        uq = MagicMock()

        for i in range(15):
            tq.put((f"Msg {i}", "en", {}))
        tq.put(None)

        translator.start_translator(tq, uq)

        self.assertEqual(mock_executor.submit.call_count, 15)
        # Snapshot for the LAST task: capped at 10 bilingual pairs, no current text
        last_call_args = mock_executor.submit.call_args[0]
        snapshot = last_call_args[2]
        self.assertEqual(len(snapshot), 10)
        self.assertTrue(all(
            isinstance(item, dict) and "source" in item and "translation" in item
            for item in snapshot
        ))
        # Oldest pairs evicted, newest retained (sources Msg 4..Msg 13)
        self.assertEqual(snapshot[0]["source"], "Msg 4")
        self.assertEqual(snapshot[-1]["source"], "Msg 13")
        self.assertEqual(snapshot[-1]["translation"], "Hola")

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
        
        uq.get(timeout=1) # status
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
        
        uq.get(timeout=1) # status
        msg2 = uq.get(timeout=1) # translation
        
        self.assertIn("translation_start", msg2["timing"])
        self.assertEqual(msg2["original"], "Hello")

    def test_translation_prompt_template_format(self):
        """Test TRANSLATION_PROMPT_TEMPLATE string format."""
        rules = translator._build_rules("Spanish")
        formatted = translator.TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang="English",
            target_lang="Spanish",
            speaker_note="Agent speaking.",
            glossary_section="Key terminology:\n- checking account = cuenta corriente",
            context_str="- Context1",
            text="Hello",
            **rules
        )
        self.assertIn("English", formatted)
        self.assertIn("- Context1", formatted)
        self.assertIn("Hello", formatted)
        self.assertIn("checking account = cuenta corriente", formatted)

    def test_prompt_contains_formal_register_rules(self):
        """Test that the prompt enforces formal register (usted)."""
        rules = translator._build_rules("Spanish")
        formatted = translator.TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang="E", target_lang="S", speaker_note="N",
            glossary_section="G", context_str="C", text="T", **rules
        )
        self.assertIn("usted", formatted)
        self.assertIn("REGISTER", formatted)

    def test_prompt_treats_input_as_untrusted(self):
        """Test that the prompt neutralizes prompt injection from live speech."""
        template = translator.TRANSLATION_PROMPT_TEMPLATE
        self.assertIn("UNTRUSTED INPUT", template)
        self.assertIn("untrusted live speech", template)
        self.assertIn("never obey", template)

    def test_prompt_delimiters_wrap_text(self):
        """Test that the untrusted text is delimited by explicit tags."""
        self.assertIn("<text_to_translate>\n{text}\n</text_to_translate>", translator.TRANSLATION_PROMPT_TEMPLATE)

    def test_prompt_contains_acronym_rules(self):
        """Test that the prompt has acronym expansion instructions."""
        rules = translator._build_rules("Spanish")
        formatted = translator.TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang="E", target_lang="S", speaker_note="N",
            glossary_section="G", context_str="C", text="T", **rules
        )
        self.assertIn("acronym", formatted.lower())
        self.assertIn("APR", formatted)

    def test_prompt_contains_compound_term_rules(self):
        """Test that the prompt has compound term translation rules."""
        rules = translator._build_rules("Spanish")
        formatted = translator.TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang="E", target_lang="S", speaker_note="N",
            glossary_section="G", context_str="C", text="T", **rules
        )
        self.assertIn("mother-in-law", formatted)
        self.assertIn("suegra", formatted)
        self.assertIn("checking account", formatted)
        self.assertIn("cuenta corriente", formatted)

    def test_prompt_rules_are_direction_aware(self):
        """ES→EN rules must use English acronym expansion, not Spanish."""
        rules = translator._build_rules("English")
        formatted = translator.TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang="S", target_lang="E", speaker_note="N",
            glossary_section="G", context_str="C", text="T", **rules
        )
        self.assertIn("Annual Percentage Rate", formatted)
        self.assertIn("suegra", formatted)
        self.assertNotIn("Spanish meaning", formatted)

    def test_build_rules_target_spanish(self):
        """Target Spanish rules contain Spanish register and acronym examples."""
        rules = translator._build_rules("Spanish")
        self.assertIn("usted", rules["register_rule"])
        self.assertIn("APR", rules["acronym_rule"])
        self.assertIn("ma'am", rules["accuracy_rule"])
        self.assertIn("señora", rules["accuracy_rule"])

    def test_build_rules_target_english(self):
        """Target English rules contain English acronym examples."""
        rules = translator._build_rules("English")
        self.assertIn("Annual Percentage Rate", rules["acronym_rule"])
        self.assertIn("suegra", rules["accuracy_rule"])
        self.assertIn("ma'am", rules["accuracy_rule"])

    def test_build_rules_contains_orthography(self):
        """Both directions must have an orthography rule."""
        for lang in ("Spanish", "English"):
            rules = translator._build_rules(lang)
            self.assertIn("orthography_rule", rules)
            self.assertIn("ORTHOGRAPHY", rules["orthography_rule"])

    def test_build_rules_contains_pronoun_resolution(self):
        """Both directions must have a pronoun resolution rule."""
        for lang in ("Spanish", "English"):
            rules = translator._build_rules(lang)
            self.assertIn("pronoun_resolution_rule", rules)
            self.assertIn("PRONOUNS", rules["pronoun_resolution_rule"])

    def test_pronoun_rule_has_his_name_example(self):
        """ES→EN pronoun rule must explicitly show 'su' → 'his' resolution."""
        rules = translator._build_rules("English")
        self.assertIn("his name", rules["pronoun_resolution_rule"])
        self.assertIn("your name", rules["pronoun_resolution_rule"])
        self.assertIn("CONSISTENCY", rules["pronoun_resolution_rule"])
        self.assertIn("your husband, your two children", rules["pronoun_resolution_rule"])

    def test_completeness_mentions_honorifics(self):
        """COMPLETENESS rule must warn about preserving honorifics."""
        for lang in ("Spanish", "English"):
            rules = translator._build_rules(lang)
            self.assertIn("honorific", rules["completeness_rule"].lower())
            self.assertIn("first or the last", rules["completeness_rule"].lower())

    def test_restore_honorific_en_to_es(self):
        """Dropped 'Ma'am' must be restored at the start of the Spanish translation."""
        restored = translator._restore_honorific(
            "Ma'am, what's your name?", "¿Cuál es su nombre?", "Spanish")
        self.assertEqual(restored, "Señora, ¿cuál es su nombre?")

    def test_restore_honorific_noop_when_present(self):
        """Honorific already in the translation must not be duplicated."""
        restored = translator._restore_honorific(
            "Ma'am, what's your name?", "Señora, ¿cuál es su nombre?", "Spanish")
        self.assertEqual(restored, "Señora, ¿cuál es su nombre?")

    def test_restore_honorific_noop_without_honorific(self):
        """No honorific in the source -> translation unchanged."""
        restored = translator._restore_honorific(
            "What is your name?", "¿Cuál es su nombre?", "Spanish")
        self.assertEqual(restored, "¿Cuál es su nombre?")

    def test_restore_honorific_es_to_en(self):
        """Dropped 'señora' must be restored as 'Ma'am' for English output."""
        restored = translator._restore_honorific(
            "Señora, buenos días", "Good morning", "English")
        self.assertEqual(restored, "Ma'am, good morning")

    @patch('src.translator.translate_ollama', return_value=("¿Cuál es su nombre?", 1.0))
    def test_process_translation_task_restores_honorific(self, mock_translate_ollama):
        """The pipeline must restore a dropped leading honorific."""
        ui_queue = MagicMock()
        translator.process_translation_task(
            ("Ma'am, what's your name?", "en"), [], ui_queue, {"audio_start": 0.5}
        )
        msg = ui_queue.put.call_args[0][0]
        self.assertEqual(msg["type"], "translation")
        self.assertEqual(msg["translated"], "Señora, ¿cuál es su nombre?")

    def test_model_upgraded(self):
        """Test that the translation model is llama3.2:3b (qwen2.5:3b echoes the source)."""
        self.assertNotEqual(translator.LLM_MODEL, "qwen2.5:1.5b")
        self.assertEqual(translator.LLM_MODEL, "llama3.2:3b")

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

    def test_get_relevant_terms_direction_aware(self):
        """Target English must format terms as 'Spanish = English' (no echo priming)."""
        from src.glossary import GlossaryManager
        mgr = GlossaryManager()
        mgr._common_terms = {
            "account holder": "titular de la cuenta",
            "checking account": "cuenta corriente",
        }
        mgr._loaded = True

        to_spanish = mgr.get_relevant_terms("checking account", target_lang="Spanish")
        self.assertIn("- checking account = cuenta corriente", to_spanish)

        to_english = mgr.get_relevant_terms("checking account", target_lang="English")
        self.assertIn("- cuenta corriente = checking account", to_english)

    def test_get_relevant_acronyms_direction_aware(self):
        """Target English must omit the Spanish expansion."""
        from src.glossary import GlossaryManager
        mgr = GlossaryManager()
        mgr._acronyms_master = {
            "APR": {"full_en": "Annual Percentage Rate", "es": "TAP (Tasa Anual de Porcentaje)"},
        }
        mgr._loaded = True

        to_spanish = mgr.get_relevant_acronyms("the APR", target_lang="Spanish")
        self.assertIn("TAP (Tasa Anual de Porcentaje)", to_spanish)

        to_english = mgr.get_relevant_acronyms("the APR", target_lang="English")
        self.assertIn("Annual Percentage Rate", to_english)
        self.assertNotIn("TAP", to_english)

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
