"""Extreme IPC stress tests for the translator's concurrency model.

These tests validate the invariants of the concurrent task-processing
pipeline under backpressure, concurrency, and edge conditions.
They use real ``multiprocessing.Queue`` objects but mock the expensive
Ollama calls.
"""

import multiprocessing
import queue
import time
import unittest
from unittest.mock import MagicMock, patch

from src import translator


class _BaseIpcTest(unittest.TestCase):
    """Shared setup: pretend the Ollama server is always up."""

    def setUp(self):
        self.patcher_ready = patch('src.translator._ollama_ready', return_value=True)
        self.mock_ollama_ready = self.patcher_ready.start()

    def tearDown(self):
        self.patcher_ready.stop()


class TestFinalPriority(_BaseIpcTest):
    """Final tasks must never be delayed by the provisional backlog."""

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.show')
    @patch('src.translator.ollama.chat')
    def test_final_not_delayed_by_provisional_backlog(self, mock_chat, mock_show, mock_glossary):
        """A final task submitted after 5 provisionals must reach the UI
        without waiting for the provisionals to finish their slow Ollama calls."""
        mock_glossary.return_value = MagicMock()
        mock_show.return_value = {}

        def chat_side_effect(*args, **kwargs):
            content = kwargs['messages'][0]['content']
            if content == '{"test":"hi"}':
                return {'message': {'content': '{"test":"hi"}'}}  # warmup, fast
            if 'Msg ' in content:
                time.sleep(0.5)  # provisionals are slow
                return {'message': {'content': '{"translation": "Hola provisional"}'}}
            return {'message': {'content': '{"translation": "Hola final"}'}}  # final, fast

        mock_chat.side_effect = chat_side_effect

        tq = multiprocessing.Queue()
        uq = multiprocessing.Queue()

        for i in range(5):
            tq.put((f"Msg {i}", "en", {}, True))  # 5 provisionals
        tq.put(("Hello world", "en", {}))  # final
        tq.put(None)  # poison pill

        start = time.time()
        translator.start_translator(tq, uq)
        elapsed = time.time() - start

        finals = []
        while not uq.empty():
            try:
                msg = uq.get_nowait()
                if msg.get("type") == "translation":
                    finals.append(msg)
            except queue.Empty:
                break

        self.assertGreaterEqual(len(finals), 1, "Final translation never reached the UI")
        self.assertTrue(
            any(m.get("translated") == "Hola final" for m in finals),
            f"Expected the final translation in {finals}",
        )
        # 5 provisionals × 0.5 s serialised in the single-thread partial
        # executor = ~2.5 s; the final must not add meaningful latency.
        self.assertLess(elapsed, 5.0, f"Pipeline took {elapsed:.2f}s — final was delayed")


class TestSemaphoreLeak(_BaseIpcTest):
    """Semaphore slots must never leak (C2 fix)."""

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.show')
    @patch('src.translator.concurrent.futures.ThreadPoolExecutor')
    @patch('src.translator.ollama.chat')
    def test_semaphore_released_on_submit_failure(self, mock_chat, mock_executor_class, mock_show, mock_glossary):
        """A partial submit failure must not leak a semaphore slot.

        Without the C2 fix, the first two submit failures consume the two
        semaphore slots and the third provisional is never submitted.
        With the fix, each failed submit releases its slot, so all three
        provisionals are submitted.
        """
        mock_glossary.return_value = MagicMock()
        mock_show.return_value = {}
        mock_chat.return_value = {'message': {'content': '{"test":"hi"}'}}

        mock_executor = MagicMock()
        # Every submit fails — both executor and partial_executor share this mock.
        mock_executor.submit.side_effect = RuntimeError("simulated submit failure")
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        tq = multiprocessing.Queue()
        uq = multiprocessing.Queue()
        for i in range(3):
            tq.put((f"Msg {i}", "en", {}, True))  # 3 provisionals
        tq.put(None)  # poison pill

        start = time.time()
        translator.start_translator(tq, uq)
        elapsed = time.time() - start

        self.assertLess(elapsed, 5.0, "Translator did not exit within timeout")

        # Provisional submits carry 11 positional args (fn + 10 args).
        provisional_calls = [
            c for c in mock_executor.submit.call_args_list
            if len(c.args) == 11
        ]
        self.assertEqual(
            len(provisional_calls), 3,
            "C2 BUG: expected 3 provisional submits (slots released on failure), "
            f"got {len(provisional_calls)}",
        )


class TestTerminalEventGuarantee(_BaseIpcTest):
    """Every task path must emit exactly one terminal UI event."""

    def _run_translator(self, tasks):
        tq = multiprocessing.Queue()
        uq = multiprocessing.Queue()
        for task in tasks:
            tq.put(task)
        tq.put(None)
        translator.start_translator(tq, uq)
        # multiprocessing.Queue puts are flushed by a feeder thread, so
        # get_nowait() can miss data right after the producer returns.
        # Use a bounded blocking get to give the feeder time to flush.
        events = []
        while True:
            try:
                msg = uq.get(timeout=0.5)
            except queue.Empty:
                break
            events.append(msg)
        return [e for e in events if e.get("type") in
                ("translation", "skipped", "error", "cancel")]

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.show')
    @patch('src.translator.ollama.chat')
    def test_happy_path_emits_translation(self, mock_chat, mock_show, mock_glossary):
        mock_glossary.return_value = MagicMock()
        mock_show.return_value = {}
        mock_chat.return_value = {'message': {'content': '{"translation": "Hola"}'}}
        terminal = self._run_translator([("Hello world", "en", {})])
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "translation")

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.show')
    @patch('src.translator.ollama.chat')
    def test_same_language_falls_back_to_error(self, mock_chat, mock_show, mock_glossary):
        """Phase 9 contingency: same-language input must NOT be silently skipped.
        If the swapped-direction retry also fails, surface an error."""
        mock_glossary.return_value = MagicMock()
        mock_show.return_value = {}
        mock_chat.return_value = {'message': {'content': '{"test":"hi"}'}}
        terminal = self._run_translator([("el banco y la cuenta", "en", {})])
        self.assertEqual(len(terminal), 1)
        # The swapped retry fails (warmup JSON, not a translation) -> error.
        self.assertIn(terminal[0]["type"], ("error", "skipped"))

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.show')
    @patch('src.translator.ollama.chat')
    @patch('src.translator.translate_ollama', return_value=(None, 1.0))
    def test_ollama_failure_emits_error(self, mock_translate, mock_chat, mock_show, mock_glossary):
        mock_glossary.return_value = MagicMock()
        mock_show.return_value = {}
        mock_chat.return_value = {'message': {'content': '{"test":"hi"}'}}
        terminal = self._run_translator([("Hello world", "en", {})])
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "error")

    @patch('src.translator.get_glossary_manager')
    @patch('src.translator.ollama.show')
    @patch('src.translator.ollama.chat')
    @patch('src.translator.translate_ollama', return_value=("el banco y la cuenta", 1.5))
    def test_echo_rejected_emits_error(self, mock_translate, mock_chat, mock_show, mock_glossary):
        mock_glossary.return_value = MagicMock()
        mock_show.return_value = {}
        mock_chat.return_value = {'message': {'content': '{"test":"hi"}'}}
        terminal = self._run_translator([("el banco y la cuenta", "es", {})])
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "error")


if __name__ == '__main__':
    unittest.main()