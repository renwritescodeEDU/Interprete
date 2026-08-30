"""
Glossary system for professional interpretation.

Loads industry-specific glossaries from JSON files, detects the conversation
industry from context, and extracts relevant terms/acronyms to inject into
the translation prompt for accurate, domain-aware translations.
"""

import json
import os
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the glossary directory
GLOSSARIES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "protocols", "glossaries"
)

# Maximum number of glossary terms to inject into a single prompt
MAX_TERMS_IN_PROMPT = 40
# Maximum number of acronyms to inject into a single prompt
MAX_ACRONYMS_IN_PROMPT = 20
# Minimum keyword matches to consider an industry detected
MIN_KEYWORD_MATCHES = 2


class GlossaryManager:
    """Manages loading, caching, and querying of industry glossaries."""

    def __init__(self, glossaries_dir: str = GLOSSARIES_DIR):
        self.glossaries_dir = glossaries_dir
        self._glossaries: dict = {}           # industry_id -> glossary data
        self._acronyms_master: dict = {}      # master acronyms lookup
        self._common_terms: dict = {}         # common/universal terms
        self._loaded = False
        self._current_industry: Optional[str] = None  # sticky industry for the session
        # Precomputed lowercase lookups (built by load_all) so the hot
        # matching paths never re-lowercase terms or keywords per call.
        self._industry_kw_en_lower: dict = {}   # industry_id -> [lowercased en keywords]
        self._industry_kw_es_lower: dict = {}   # industry_id -> [lowercased es keywords]
        self._industry_terms_lower: dict = {}   # industry_id -> [(lower, original, es)]

    def _build_terms_lower(self, terms: dict):
        """[(lower, original, es)] preserving insertion order of ``terms``."""
        return [(term.lower(), term, es) for term, es in terms.items()]

    def load_all(self) -> None:
        """Load all glossary JSON files from the glossaries directory."""
        if self._loaded:
            return

        if not os.path.isdir(self.glossaries_dir):
            logger.warning(f"Glossaries directory not found: {self.glossaries_dir}")
            self._loaded = True
            return

        loaded_count = 0
        for filename in os.listdir(self.glossaries_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.glossaries_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                industry_id = data.get("industry_id", filename.replace(".json", ""))

                if industry_id == "acronyms_master":
                    self._acronyms_master = data.get("acronyms", {})
                    logger.info(f"Loaded {len(self._acronyms_master)} master acronyms")
                elif industry_id == "common":
                    self._common_terms = data.get("terms", {})
                    logger.info(f"Loaded {len(self._common_terms)} common terms")
                else:
                    self._glossaries[industry_id] = data
                    term_count = len(data.get("terms", {}))
                    logger.info(f"Loaded glossary '{industry_id}' with {term_count} terms")
                    self._industry_kw_en_lower[industry_id] = [
                        kw.lower() for kw in data.get("detection_keywords_en", [])
                    ]
                    self._industry_kw_es_lower[industry_id] = [
                        kw.lower() for kw in data.get("detection_keywords_es", [])
                    ]
                    self._industry_terms_lower[industry_id] = self._build_terms_lower(
                        data.get("terms", {})
                    )

                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load glossary {filename}: {e}")

        self._loaded = True
        logger.info(f"Glossary system initialized: {loaded_count} files, "
                     f"{len(self._glossaries)} industries, "
                     f"{len(self._acronyms_master)} acronyms, "
                     f"{len(self._common_terms)} common terms")

    def detect_industry(self, context_texts: list[str]) -> Optional[str]:
        """
        Detect the most likely industry from conversation context.
        Uses sticky detection: once an industry is identified it remains
        active for the entire session unless a stronger signal overrides it.
        """
        if not self._glossaries:
            return self._current_industry

        combined_text = " ".join(context_texts).lower()
        scores: dict[str, int] = {}

        for industry_id, glossary in self._glossaries.items():
            score = 0
            for kw in self._industry_kw_en_lower.get(industry_id, []):
                if kw in combined_text:
                    score += 1
            for kw in self._industry_kw_es_lower.get(industry_id, []):
                if kw in combined_text:
                    score += 1

            # Accept even 1 keyword match when no industry is yet known (first phrase)
            threshold = 1 if self._current_industry is None else MIN_KEYWORD_MATCHES
            if score >= threshold:
                scores[industry_id] = score

        if scores:
            best = max(scores, key=scores.get)
            if best != self._current_industry:
                logger.info(f"Industry detected/updated: {best} (score: {scores[best]})")
                self._current_industry = best

        return self._current_industry

    def reset_session(self) -> None:
        """Reset sticky industry at the start of a new call/session."""
        self._current_industry = None
        logger.info("[GLOSSARY] Session reset — industry detection cleared.")

    def get_relevant_terms(self, text: str, industry_id: Optional[str] = None,
                           target_lang: str = "Spanish",
                           max_terms: int = MAX_TERMS_IN_PROMPT) -> str:
        """
        Extract relevant glossary terms for the given input text.

        Returns a formatted string suitable for prompt injection, ordered in
        the translation direction: "English = Spanish" when translating to
        Spanish, "Spanish = English" when translating to English. The
        source->target ordering prevents small models from being primed into
        echoing the source language (e.g. a list of "English = Spanish" pairs
        makes an ES->EN model output Spanish).

        max_terms caps the injected vocabulary; short utterances should pass a
        smaller budget to keep the LLM prompt (and its processing time) small.
        """
        text_lower = text.lower()
        matched_terms: dict[str, str] = {}

        # 1. Match common terms against input text
        for en_term, es_term in self._common_terms.items():
            if en_term.lower() in text_lower:
                matched_terms[en_term] = es_term

        # 2. Match industry-specific terms against input text
        if industry_id and industry_id in self._glossaries:
            industry_terms = self._glossaries[industry_id].get("terms", {})
            for term_lower, en_term, es_term in self._industry_terms_lower.get(industry_id, []):
                if term_lower in text_lower:
                    matched_terms[en_term] = es_term

            # 3. If few matches, add some high-frequency terms from the industry
            if len(matched_terms) < 10:
                # Add first N terms from the industry as context hints
                remaining = max_terms - len(matched_terms)
                for en_term, es_term in list(industry_terms.items())[:remaining]:
                    if en_term not in matched_terms:
                        matched_terms[en_term] = es_term

        # Limit total terms
        if len(matched_terms) > max_terms:
            # Prioritize exact matches in the text
            exact_matches = {k: v for k, v in matched_terms.items() if k.lower() in text_lower}
            other_matches = {k: v for k, v in matched_terms.items() if k.lower() not in text_lower}
            matched_terms = dict(list(exact_matches.items())[:max_terms])
            remaining = max_terms - len(matched_terms)
            if remaining > 0:
                matched_terms.update(dict(list(other_matches.items())[:remaining]))

        if not matched_terms:
            return ""

        if target_lang == "English":
            lines = [f"- {es} = {en}" for en, es in matched_terms.items()]
        else:
            lines = [f"- {en} = {es}" for en, es in matched_terms.items()]
        return "\n".join(lines)

    def get_relevant_acronyms(self, text: str, industry_id: Optional[str] = None,
                              target_lang: str = "Spanish") -> str:
        """
        Extract relevant acronyms for the given input text.

        When the target is English the Spanish expansion is omitted so the
        model is not primed into outputting Spanish.
        """
        # Find potential acronyms in the text (2-6 uppercase letters)
        potential_acronyms = set(re.findall(r'\b[A-Z]{2,6}\b', text))

        # Also check for common acronyms that might appear in lowercase or mixed context
        text_upper = text.upper()

        matched_acronyms: dict[str, dict] = {}

        # Check master acronyms
        for acronym, data in self._acronyms_master.items():
            if acronym in potential_acronyms or acronym in text_upper:
                matched_acronyms[acronym] = data

        # Check industry-specific acronyms
        if industry_id and industry_id in self._glossaries:
            industry_acronyms = self._glossaries[industry_id].get("acronyms", {})
            for acronym, data in industry_acronyms.items():
                if acronym in potential_acronyms or acronym in text_upper:
                    matched_acronyms[acronym] = data

        # Also check if the full English name of any acronym appears in the text
        text_lower = text.lower()
        for acronym, data in self._acronyms_master.items():
            full_en = data.get("full_en", "").lower()
            if full_en and full_en in text_lower:
                matched_acronyms[acronym] = data

        # Limit
        if len(matched_acronyms) > MAX_ACRONYMS_IN_PROMPT:
            matched_acronyms = dict(list(matched_acronyms.items())[:MAX_ACRONYMS_IN_PROMPT])

        if not matched_acronyms:
            return ""

        lines = []
        for acronym, data in matched_acronyms.items():
            full_en = data.get("full_en", "")
            es = data.get("es", "")
            if target_lang == "English":
                lines.append(f"- {acronym} ({full_en})")
            else:
                lines.append(f"- {acronym} ({full_en}) → {es}")

        return "\n".join(lines)

    def build_glossary_prompt_section(self, text: str, context_history: list[str],
                                      target_lang: str = "Spanish",
                                      max_terms: int = MAX_TERMS_IN_PROMPT) -> str:
        """
        Build the complete glossary section for the translation prompt.

        This is the main entry point called by the translator.
        It detects the industry, finds relevant terms and acronyms,
        and formats them for injection into the prompt, ordered in the
        translation direction (target_lang).
        """
        self.load_all()

        # Detect industry from all available context
        all_context = context_history + [text]
        industry_id = self.detect_industry(all_context)

        # Get relevant terms and acronyms
        terms_section = self.get_relevant_terms(text, industry_id, target_lang, max_terms)
        acronyms_section = self.get_relevant_acronyms(text, industry_id, target_lang)

        parts = []

        if industry_id and industry_id in self._glossaries:
            display_name = self._glossaries[industry_id].get("display_name", industry_id)
            parts.append(f"[Industry: {display_name}]")

        if terms_section:
            parts.append(f"Key terminology:\n{terms_section}")

        if acronyms_section:
            if target_lang == "English":
                parts.append(f"Acronyms (use English meaning):\n{acronyms_section}")
            else:
                parts.append(f"Acronyms (use Spanish equivalent with meaning in parentheses):\n{acronyms_section}")

        if not parts:
            return "No specific terminology loaded."

        return "\n\n".join(parts)


# Module-level singleton for the glossary manager
_manager: Optional[GlossaryManager] = None


def get_glossary_manager() -> GlossaryManager:
    """Get or create the singleton GlossaryManager instance."""
    global _manager
    if _manager is None:
        _manager = GlossaryManager()
    return _manager
