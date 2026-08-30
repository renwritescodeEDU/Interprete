"""Characterization tests for best-effort queue delivery helpers.

Each worker module exposes its own put helper with a specific contract.
These tests pin those contracts so the shared implementation in
queueutil.py (Lote C) can replace the bodies without changing behavior.
"""

import queue
import unittest
from unittest.mock import MagicMock, patch

from src.audio import _safe_ui_put
from src.transcriber import _send_to_queue
from src.translator import _put_ui


class TestSendToQueue(unittest.TestCase):
    def test_success_returns_true(self):
        q = queue.Queue()
        self.assertTrue(_send_to_queue(q, "msg"))

    def test_full_blocking_logs_error(self):
        q = queue.Queue(maxsize=1)
        q.put("first")
        with patch("src.queueutil.logger.error") as mock_error:
            self.assertFalse(_send_to_queue(q, "second", block=True, timeout=0.01, error_msg="Queue is full"))
        mock_error.assert_called_once_with("Queue is full")

    def test_full_non_blocking_is_silent(self):
        q = queue.Queue(maxsize=1)
        q.put("first")
        with patch("src.transcriber.logger.error") as mock_error:
            self.assertFalse(_send_to_queue(q, "second", block=False))
        mock_error.assert_not_called()

    def test_generic_exception_logs_debug(self):
        mock_q = MagicMock()
        mock_q.put.side_effect = RuntimeError("boom")
        with patch("src.queueutil.logger.debug") as mock_debug:
            self.assertFalse(_send_to_queue(mock_q, "msg"))
        mock_debug.assert_called_once()
        self.assertIn("Queue communication error", mock_debug.call_args[0][0])

    def test_put_with_timeout_none(self):
        q = queue.Queue()
        self.assertTrue(_send_to_queue(q, "msg", timeout=None))


class TestPutUi(unittest.TestCase):
    def test_success_silent(self):
        mock_q = MagicMock()
        self.assertIsNone(_put_ui(mock_q, {"type": "x"}, timeout=1.0))
        mock_q.put.assert_called_once()

    def test_failure_logs_debug(self):
        mock_q = MagicMock()
        mock_q.put.side_effect = queue.Full
        with patch("src.queueutil.logger.debug") as mock_debug:
            self.assertIsNone(_put_ui(mock_q, {"type": "x"}, timeout=1.0))
        mock_debug.assert_called_once()
        self.assertIn("[TRANSLATOR] ui_queue put failed", mock_debug.call_args[0][0])


class TestSafeUiPut(unittest.TestCase):
    def test_none_queue_noop(self):
        self.assertIsNone(_safe_ui_put(None, {"type": "x"}))

    def test_success(self):
        mock_q = MagicMock()
        self.assertIsNone(_safe_ui_put(mock_q, {"type": "x"}))
        mock_q.put.assert_called_once()

    def test_failure_is_silent(self):
        mock_q = MagicMock()
        mock_q.put.side_effect = Exception("boom")
        _safe_ui_put(mock_q, {"type": "x"})
        # No exception raised and nothing logged — that is the contract.


if __name__ == "__main__":
    unittest.main()