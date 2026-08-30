"""Centralized contracts for inter-process UI events (Phase 6.1).

All worker processes deliver UI events through ``multiprocessing.Queue``
as plain dicts. This module is the single source of truth for their shape:

* the builders below produce exactly the dicts the UI consumes, so a new
  message type is added in one place and validated by tests;
* the pipeline ``timing`` dict keys are named constants so no module
  relies on string literals.

The serialized format is unchanged from the pre-Phase-6 implementation:
the UI only needs to keep reading the same shapes.
"""

# --- Pipeline timing dict keys ---
TIMING_RECORDING_START = "recording_start"
TIMING_RECORDING_STOP = "recording_stop"
TIMING_TRANSCRIPTION_START = "transcription_start"
TIMING_TRANSCRIPTION_END = "transcription_end"
TIMING_TRANSLATION_START = "translation_start"
TIMING_TRANSLATION_END = "translation_end"

# --- Event type names ---
TYPE_STATUS = "status"
TYPE_PARTIAL = "partial"
TYPE_FINAL = "final"
TYPE_PROVISIONAL = "provisional"
TYPE_TRANSLATION = "translation"
TYPE_CANCEL = "cancel"
TYPE_SKIPPED = "skipped"
TYPE_TRUNCATED = "truncated"
TYPE_ERROR = "error"

# --- Skip reasons ---
SKIP_REASON_SAME_LANGUAGE = "same_language"
SKIP_REASON_QUEUE_FULL = "queue_full"
SKIP_STAGE_UI = "ui"
SKIP_STAGE_TRANSLATION = "translation"

# --- Cancel reasons ---
CANCEL_REASON_EMPTY_AUDIO = "empty_audio"
CANCEL_REASON_NO_SPEECH = "no_speech"

# --- Status values ---
STATUS_READY = "ready"
STATUS_WAITING = "waiting"
STATUS_OLLAMA_WAITING = "ollama_waiting"
STATUS_OLLAMA_OFFLINE = "ollama_offline"
STATUS_MODEL_DOWNLOAD = "model_download"

# --- Process names ---
PROCESS_AUDIO = "audio"
PROCESS_TRANSCRIBER = "transcriber"
PROCESS_TRANSLATOR = "translator"

# The terminal events the UI treats as "a STOP produced a result".
TERMINAL_EVENT_TYPES = (TYPE_TRANSLATION, TYPE_SKIPPED, TYPE_ERROR, TYPE_CANCEL)


def ui_status(process: str, status: str, model: str = None) -> dict:
    msg = {"type": TYPE_STATUS, "process": process, "status": status}
    if model is not None:
        msg["model"] = model
    return msg


def ui_partial(text: str) -> dict:
    return {"type": TYPE_PARTIAL, "text": text}


def ui_final(text: str) -> dict:
    return {"type": TYPE_FINAL, "text": text}


def ui_provisional(original: str, translated: str) -> dict:
    return {"type": TYPE_PROVISIONAL, "original": original, "translated": translated}


def ui_translation(original: str, translated: str, latency: float, timing: dict) -> dict:
    return {
        "type": TYPE_TRANSLATION,
        "original": original,
        "translated": translated,
        "latency": latency,
        "timing": timing,
    }


def ui_cancel(reason: str) -> dict:
    return {"type": TYPE_CANCEL, "reason": reason}


def ui_skipped(reason: str, original: str = None, stage: str = None) -> dict:
    msg = {"type": TYPE_SKIPPED, "reason": reason}
    if original is not None:
        msg["original"] = original
    if stage is not None:
        msg["stage"] = stage
    return msg


def ui_truncated(dropped_seconds: float, max_minutes: int) -> dict:
    return {
        "type": TYPE_TRUNCATED,
        "dropped_seconds": dropped_seconds,
        "max_minutes": max_minutes,
    }


def ui_error(message: str) -> dict:
    return {"type": TYPE_ERROR, "message": message}


def is_terminal_event(msg: dict) -> bool:
    """True when the message is one of the terminal UI events."""
    return isinstance(msg, dict) and msg.get("type") in TERMINAL_EVENT_TYPES
