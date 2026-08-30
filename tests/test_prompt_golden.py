"""Golden tests for the translation prompt template and rules.

The prompt is the highest-leverage surface of translation quality for
llama3.2:3b. These snapshots force any change to be deliberate and
visible in code review.
"""

import json
import os
import unittest

from src.translator import TRANSLATION_PROMPT_TEMPLATE, _build_rules

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "prompt_golden.json")

SPEAKER_NOTE = "Agent or client speaking — use formal register (usted)."


def _load_golden():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestPromptGolden(unittest.TestCase):
    """Verifies the prompt template and rules match the committed snapshot."""

    def test_template_frozen(self):
        golden = _load_golden()
        self.assertEqual(TRANSLATION_PROMPT_TEMPLATE, golden["template"])

    def test_rules_spanish_frozen(self):
        golden = _load_golden()
        self.assertEqual(_build_rules("Spanish"), golden["rules_spanish"])

    def test_rules_english_frozen(self):
        golden = _load_golden()
        self.assertEqual(_build_rules("English"), golden["rules_english"])

    def test_formatted_prompt_spanish_frozen(self):
        golden = _load_golden()
        formatted = TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang="English",
            target_lang="Spanish",
            speaker_note=SPEAKER_NOTE,
            glossary_section="No specific terminology loaded.",
            context_str="(No prior context)",
            text="Hello",
            **_build_rules("Spanish"),
        )
        self.assertEqual(formatted, golden["formatted_es"])

    def test_formatted_prompt_english_frozen(self):
        golden = _load_golden()
        formatted = TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang="Spanish",
            target_lang="English",
            speaker_note=SPEAKER_NOTE,
            glossary_section="No specific terminology loaded.",
            context_str="(No prior context)",
            text="Hola",
            **_build_rules("English"),
        )
        self.assertEqual(formatted, golden["formatted_en"])


if __name__ == "__main__":
    unittest.main()