"""Characterization tests for glossary output.

The golden file at tests/golden/glossary_golden.json records the exact
output produced by the current glossary implementation. Any refactoring
of the glossary logic must keep these outputs byte-identical.

To regenerate the golden file:
  python -m tests.golden.regenerate  (TBD)
"""

import json
import os
import unittest

from src.glossary import GlossaryManager

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "glossary_golden.json")


def _fresh_manager():
    mgr = GlossaryManager()
    mgr.load_all()
    mgr.reset_session()
    return mgr


def _load_golden():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestGlossaryGolden(unittest.TestCase):
    """Verifies glossary outputs match the committed golden snapshot."""

    def test_build_sections(self):
        golden = _load_golden()
        for key, entry in golden["build_sections"].items():
            mgr = _fresh_manager()
            result = mgr.build_glossary_prompt_section(
                entry["text"], [], entry["lang"], entry["max_terms"]
            )
            self.assertEqual(result, entry["output"], key)

    def test_get_relevant_terms(self):
        golden = _load_golden()
        for key, entry in golden["terms"].items():
            mgr = _fresh_manager()
            industry = entry["industry"]
            max_terms = entry.get("max_terms")
            if industry:
                if max_terms is not None:
                    result = mgr.get_relevant_terms(entry["text"], industry, entry["lang"], max_terms)
                else:
                    result = mgr.get_relevant_terms(entry["text"], industry, entry["lang"])
            else:
                result = mgr.get_relevant_terms(entry["text"], None, entry["lang"])
            self.assertEqual(result, entry["output"], key)

    def test_get_relevant_acronyms(self):
        golden = _load_golden()
        for key, entry in golden["acronyms"].items():
            mgr = _fresh_manager()
            result = mgr.get_relevant_acronyms(entry["text"], entry["industry"], entry["lang"])
            self.assertEqual(result, entry["output"], key)