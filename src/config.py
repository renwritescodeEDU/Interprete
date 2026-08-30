"""Shared preferences persistence (Phase 7).

The UI writes ``.config/preferences.json``; the worker processes read it to
honour user model/device overrides. Kept here so every process shares one
implementation instead of duplicating it in ui.py.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "preferences.json")


def load_preferences(config_file: str = CONFIG_FILE) -> dict:
    """Load saved preferences from disk. Never raises — {} on any failure."""
    try:
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
    return {}


def save_preferences(config: dict, config_file: str = CONFIG_FILE) -> None:
    """Save preferences to disk. Never raises."""
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save config: {e}")
