"""Contract tests for the centralized UI-event builders (Phase 6.1).

Every event type produced by the three worker processes must carry the
exact fields the UI consumes. These tests pin the builders in
``src/events.py`` so the producers and the UI cannot drift apart.
"""

import unittest

from src.events import (
    TERMINAL_EVENT_TYPES,
    TYPE_CANCEL,
    TYPE_ERROR,
    TYPE_FINAL,
    TYPE_PARTIAL,
    TYPE_PROVISIONAL,
    TYPE_SKIPPED,
    TYPE_STATUS,
    TYPE_TRANSLATION,
    TYPE_TRUNCATED,
    is_terminal_event,
    ui_cancel,
    ui_error,
    ui_final,
    ui_partial,
    ui_provisional,
    ui_skipped,
    ui_status,
    ui_translation,
    ui_truncated,
)


class TestEventBuilders(unittest.TestCase):
    def test_ui_status_shape(self):
        msg = ui_status("audio", "ready")
        self.assertEqual(msg, {"type": TYPE_STATUS, "process": "audio", "status": "ready"})

    def test_ui_status_with_model(self):
        msg = ui_status("translator", "model_download", model="llama3.2:3b")
        self.assertEqual(
            msg,
            {"type": TYPE_STATUS, "process": "translator", "status": "model_download", "model": "llama3.2:3b"},
        )

    def test_ui_status_without_model_has_no_key(self):
        msg = ui_status("translator", "ready")
        self.assertNotIn("model", msg)

    def test_ui_partial_shape(self):
        self.assertEqual(ui_partial("Hola"), {"type": TYPE_PARTIAL, "text": "Hola"})

    def test_ui_final_shape(self):
        self.assertEqual(ui_final("Hola"), {"type": TYPE_FINAL, "text": "Hola"})

    def test_ui_provisional_shape(self):
        self.assertEqual(
            ui_provisional("Buenos días", "Good morning"),
            {"type": TYPE_PROVISIONAL, "original": "Buenos días", "translated": "Good morning"},
        )

    def test_ui_translation_shape(self):
        timing = {"recording_start": 1.0}
        self.assertEqual(
            ui_translation("Hello", "Hola", 1.5, timing),
            {"type": TYPE_TRANSLATION, "original": "Hello", "translated": "Hola",
             "latency": 1.5, "timing": timing},
        )

    def test_ui_cancel_shape(self):
        self.assertEqual(ui_cancel("no_speech"), {"type": TYPE_CANCEL, "reason": "no_speech"})

    def test_ui_skipped_minimal(self):
        self.assertEqual(ui_skipped("same_language"), {"type": TYPE_SKIPPED, "reason": "same_language"})

    def test_ui_skipped_with_original_and_stage(self):
        self.assertEqual(
            ui_skipped("queue_full", original="Hi", stage="translation"),
            {"type": TYPE_SKIPPED, "reason": "queue_full", "original": "Hi", "stage": "translation"},
        )

    def test_ui_truncated_shape(self):
        self.assertEqual(
            ui_truncated(12.5, 5),
            {"type": TYPE_TRUNCATED, "dropped_seconds": 12.5, "max_minutes": 5},
        )

    def test_ui_error_shape(self):
        self.assertEqual(
            ui_error("Translation Error: x"),
            {"type": TYPE_ERROR, "message": "Translation Error: x"},
        )


class TestTerminalEvents(unittest.TestCase):
    def test_terminal_event_types_include_all_terminal_paths(self):
        self.assertEqual(
            set(TERMINAL_EVENT_TYPES),
            {TYPE_TRANSLATION, TYPE_SKIPPED, TYPE_ERROR, TYPE_CANCEL},
        )

    def test_is_terminal_event(self):
        self.assertTrue(is_terminal_event({"type": TYPE_TRANSLATION}))
        self.assertTrue(is_terminal_event({"type": TYPE_SKIPPED}))
        self.assertTrue(is_terminal_event({"type": TYPE_ERROR}))
        self.assertTrue(is_terminal_event({"type": TYPE_CANCEL}))
        self.assertFalse(is_terminal_event({"type": TYPE_PARTIAL}))
        self.assertFalse(is_terminal_event({"type": TYPE_STATUS}))
        self.assertFalse(is_terminal_event({}))
        self.assertFalse(is_terminal_event(None))


if __name__ == "__main__":
    unittest.main()