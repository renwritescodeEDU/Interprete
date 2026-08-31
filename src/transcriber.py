import multiprocessing
import os
import queue
import time
import logging
from faster_whisper import WhisperModel

from src.events import (
    ui_cancel,
    ui_error,
    ui_final,
    ui_partial,
    ui_skipped,
    ui_status,
)
from src.hardware import (
    DEFAULT_WHISPER_COMPUTE,
    DEFAULT_WHISPER_MODEL,
    get_hardware_profile,
    select_whisper_config,
)
from src.queueutil import put_best_effort

logger = logging.getLogger(__name__)

# Constants (absolute CPU fallback; the real selection happens in
# `_resolve_whisper_config` / src.hardware).
MODEL_SIZE = DEFAULT_WHISPER_MODEL
COMPUTE_TYPE = DEFAULT_WHISPER_COMPUTE
DEVICE = get_hardware_profile().device
# Beam size for the FINAL transcription of a recorded utterance.
BEAM_SIZE = 5
# CPU beam sizes for long finals (Step 3): the dominant CPU cost of Whisper
# is beam search on the final chunk of a long recording (observed: 29 s on a
# single 8.5 s clip). The beam shrinks as the audio grows so stop→display
# latency stays within budget on CPU.
BEAM_SIZE_CPU_MEDIUM = 3
BEAM_SIZE_CPU_LONG = 2
# Duration thresholds (seconds) for the CPU adaptive beam.
BEAM_CPU_MEDIUM_SECONDS = 5.0
BEAM_CPU_LONG_SECONDS = 10.0
# Beam size for live PARTIAL transcripts. Beam search is the dominant CPU
# cost; a beam of 1 (greedy) on partials keeps stop→display latency within
# budget during capture, while finals still get the quality of beam 5.
BEAM_SIZE_PARTIAL = 1
# Native capture rate; used as a safe fallback when a chunk arrives without
# a usable rate value.
RATE = 16000
SUPPORTED_LANGUAGES = {"en", "es"}
TIMEOUT_PUT = 5.0
TIMEOUT_GET = 1.0
# Language confidence threshold. When a final chunk's detected language
# probability falls below this value, the system falls back to the last
# confidently-detected language (or DEFAULT_LANGUAGE) instead of discarding.
# Discarding the entire utterance because of uncertain language detection
# causes data loss — the same-language guard in the translator already
# prevents inverted translations.
LANGUAGE_CONFIDENCE_THRESHOLD = 0.60
# Fallback language when detection is uncertain.
DEFAULT_LANGUAGE = "en"
# Minimum accumulated-text growth (chars) before a provisional translation
# task is forwarded to the translator. Throttles LLM load during recording.
# With ~3s partials each chunk adds tens of characters, so a threshold of 25
# still fires on every chunk while dropping near-no-op revisions.
PARTIAL_PROGRESS_THRESHOLD = 25

# Bilingual initial prompt for partial transcriptions (Phase 8). Short
# examples of EN and ES punctuation, capitals, and question marks so
# Whisper applies correct formatting regardless of the utterance language.
# Only used for partial chunks (beam 1); finals skip the prompt to avoid
# hallucinating this text on silent audio.
TRANSCRIBE_INITIAL_PROMPT = (
    "Hello, how are you? Hola, ¿qué tal? "
    "This is a test. Esto es una prueba. "
    "I need help. Necesito ayuda."
)

def _send_to_queue(q, msg, block=False, timeout=None, error_msg="Queue put failed"):
    return put_best_effort(
        q, msg, block=block, timeout=timeout,
        error_msg=error_msg, debug_msg="Queue communication error",
    )


def _resolve_whisper_config() -> dict:
    """Resolve the whisper model/compute/device for this machine.

    Precedence: ``INTERPRETE_DEVICE`` env override > user preferences >
    hardware auto-scaling (see ``src.hardware``). Never raises.
    """
    cfg = select_whisper_config()
    requested = os.environ.get("INTERPRETE_DEVICE")
    if requested == "cpu":
        cfg = {"model": MODEL_SIZE, "compute_type": "int8", "device": "cpu"}
        logger.warning(
            "[TRANSCRIBER] INTERPRETE_DEVICE=cpu forces CPU (small/int8)."
        )
    elif requested == "cuda":
        if get_hardware_profile().device == "cuda":
            cfg["device"] = "cuda"
        else:
            cfg["device"] = "cpu"
            logger.warning(
                "[TRANSCRIBER] INTERPRETE_DEVICE=cuda set, but no usable CUDA "
                "runtime — using CPU instead."
            )
    return cfg


def _create_model():
    """Create the WhisperModel using the resolved hardware config, with a
    guaranteed CPU fallback so the pipeline never dies on model init.

    Creation on CUDA can still fail (missing runtime, driver mismatch);
    if it raises, retry once on CPU with the safe small/int8 fallback.
    """
    cfg = _resolve_whisper_config()
    if cfg["device"] == "cpu":
        logger.info(
            f"[TRANSCRIBER] Creating WhisperModel(model={cfg['model']}, "
            f"device=cpu, compute_type={cfg['compute_type']})."
        )
        return WhisperModel(cfg["model"], device="cpu", compute_type=cfg["compute_type"])
    try:
        logger.info(
            f"[TRANSCRIBER] Creating WhisperModel(model={cfg['model']}, "
            f"device=cuda, compute_type={cfg['compute_type']})."
        )
        return WhisperModel(cfg["model"], device="cuda", compute_type=cfg["compute_type"])
    except Exception as e:
        logger.warning(
            f"[TRANSCRIBER] Model creation on CUDA failed ({e}) — falling "
            f"back to CPU (model={MODEL_SIZE}, compute_type={COMPUTE_TYPE})."
        )
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)

def _join_transcript(segments, overlap_seconds: float) -> str:
    """Join transcribed segments into text, discarding the overlap window.

    When ``overlap_seconds`` is 0 (or absent), behaves exactly like
    ``"".join(segment.text ...)`` — the pre-overlap behaviour.

    When an overlap window exists (Step 5), words whose ``end`` timestamp
    falls at or before ``overlap_seconds`` are the acoustic context prepended
    from the previous segment and must NOT appear in the transcript (they
    were already delivered with the previous final). Word-level trimming is
    used when the model returned word timestamps; otherwise segments that
    lie entirely inside the window are dropped as a fallback.
    """
    if overlap_seconds <= 0:
        return "".join(segment.text for segment in segments).strip()

    parts = []
    for segment in segments:
        words = getattr(segment, "words", None)
        if isinstance(words, list) and words:
            kept = [w.word for w in words
                    if float(getattr(w, "end", 0.0)) > overlap_seconds]
            parts.extend(kept)
        else:
            # No word timestamps: skip segments fully inside the overlap.
            if float(getattr(segment, "end", 0.0)) > overlap_seconds:
                parts.append(getattr(segment, "text", ""))
    return "".join(parts).strip()


def _select_beam_size(duration_seconds: float, device: str, is_final: bool) -> int:
    """Pick the beam size for a transcription call.

    Latency policy (Step 3):
    - GPU: maximum quality (beam 5) unconditionally — GPU throughput makes
      the beam cheap, so long clips still transcribe quickly.
    - CPU: beam shrinks as the audio grows, because beam search is the
      dominant CPU cost (observed 29 s on a long clip):
        * duration < 5 s        -> beam 5 (full quality, short clips are cheap)
        * 5 s <= duration < 10 s -> beam 3 (quality/latency compromise)
        * duration >= 10 s       -> beam 2 (latency priority)
    - Partials always use the greedy beam (1) for maximum capture speed;
      only the FINAL chunk of an utterance gets the quality beam.
    """
    if not is_final:
        return BEAM_SIZE_PARTIAL
    if device == "cuda":
        return BEAM_SIZE
    if duration_seconds < BEAM_CPU_MEDIUM_SECONDS:
        return BEAM_SIZE
    if duration_seconds < BEAM_CPU_LONG_SECONDS:
        return BEAM_SIZE_CPU_MEDIUM
    return BEAM_SIZE_CPU_LONG


def start_transcriber(
    asr_queue: multiprocessing.Queue, translation_queue: multiprocessing.Queue, ui_queue: multiprocessing.Queue,
    final_queue: multiprocessing.Queue = None
):
    """
    Pulls audio chunks from asr_queue, transcribes them, and pushes (text, language, timing) tuples
    to translation_queue. Supports 3-element (legacy) and 4-element (with timing) tuples from audio.

    When final_queue is provided, the authoritative final chunk arrives there and
    is processed with priority — it is never queued behind the partial backlog.
    """
    model = _create_model()
    # Determine the effective device: faster-whisper's WhisperModel exposes
    # .device, and in tests the model is a mock with a non-string attribute.
    _cfg = _resolve_whisper_config()
    _actual_device = getattr(model, "device", None)
    if not isinstance(_actual_device, str):
        _actual_device = _cfg.get("device", "cpu")

    _send_to_queue(ui_queue, ui_status("transcriber", "ready"))

    detected_language = None
    # Language remembered across chunks and recordings. When a final chunk's
    # detection is uncertain (low probability or an unsupported language like
    # 'nn'), this value is used as the fallback instead of discarding audio.
    last_confident_language = None
    # Progressive transcription: accumulated partial text used for live
    # previews and for provisional translations while recording continues.
    provisional_text = ""
    last_provisional_sent_len = 0
    # Text of the last transcribed final, chained as context for the next
    # chunk so Whisper continues the sentence naturally (Phase 8.2).
    last_final_text = None

    while True:
        try:
            # Priority: if a final is waiting on the dedicated final queue,
            # process it immediately instead of spending ~2s on the next
            # partial. This keeps stop->display latency flat regardless of
            # the partial backlog accumulated during long recordings.
            item = None
            if final_queue is not None:
                try:
                    item = final_queue.get_nowait()
                except queue.Empty:
                    pass
            if item is None:
                item = asr_queue.get(timeout=TIMEOUT_GET)
            if item is None or item == "QUIT":
                break

            # Support both 3-element (legacy) and 4-element (with timing) tuples
            if not isinstance(item, tuple) or len(item) not in (3, 4):
                continue

            if len(item) == 4:
                audio_data, rate, is_final, timing = item
            else:
                audio_data, rate, is_final = item
                timing = {}
            logger.info(
                f"[TRANSCRIBER] Chunk received: {len(audio_data)} samples, "
                f"rate={rate}, is_final={is_final}"
            )

            # The final supersedes any stale partials still queued behind it.
            # Drain them so the authoritative transcription isn't delayed by
            # the partial backlog (observed 20s+ waits on long recordings).
            # Careful: preserve the process-termination poison pill (QUIT/None).
            if is_final:
                while True:
                    try:
                        stale = asr_queue.get_nowait()
                    except queue.Empty:
                        break
                    if stale is None or stale == "QUIT":
                        asr_queue.put(stale)
                        break
            
            # Guard clause: Empty audio
            if audio_data is None or len(audio_data) == 0:
                if is_final:
                    _send_to_queue(ui_queue, ui_cancel("empty_audio"))
                    logger.warning("[TRANSCRIBER] Final chunk was empty — sent cancel")
                continue

            transcription_start = time.time()

            # Determine initial prompt. A bilingual prompt forces Whisper to
            # use correct punctuation in both languages. When a previous final
            # exists, its text is chained as context so Whisper continues the
            # sentence naturally across chunks (Phase 8.2).
            prompt = TRANSCRIBE_INITIAL_PROMPT
            if last_final_text:
                prompt = f"{TRANSCRIBE_INITIAL_PROMPT}\n{last_final_text}"

            # Adaptive beam size: GPU keeps maximum quality; CPU shrinks the
            # beam as the audio grows to stay within the 2 s stop→display
            # latency budget (observed 29 s on a single long clip).
            duration_seconds = len(audio_data) / rate if rate else len(audio_data) / RATE
            beam_size = _select_beam_size(duration_seconds, _actual_device, is_final)

            # Audio overlap (Step 5): the tail of the previous segment is
            # prepended as acoustic context. The overlapping words must be
            # dropped from the transcript, so request word-level timestamps
            # when an overlap window is present.
            overlap_seconds = float(timing.get("overlap_seconds", 0.0) or 0.0)

            segments, info = model.transcribe(
                audio_data,
                beam_size=beam_size,
                vad_filter=True,
                initial_prompt=prompt,
                word_timestamps=is_final and overlap_seconds > 0,
                # Whisper hallucination guards: reject output on silence/noise
                # (compression ratio too high or log-prob too low) and abort
                # early instead of emitting long garbage loops (observed: a
                # 29s transcription of an 8.5s clip).
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
            )
            
            detected_language = info.language
            language_probability = info.language_probability
            logger.info(
                f"[TRANSCRIBER] Language detected: '{detected_language}' "
                f"(p={language_probability:.2f})"
            )

            # Remember confidently-detected supported languages so an uncertain
            # final chunk can fall back instead of being discarded.
            if (detected_language in SUPPORTED_LANGUAGES
                    and language_probability >= LANGUAGE_CONFIDENCE_THRESHOLD):
                last_confident_language = detected_language

            # Unsupported language (e.g. 'nn' on degraded audio): fall back to
            # the last confident language instead of cancelling the recording.
            # Whisper routinely misdetects short/noisy clips as low-resource
            # languages ('nn', 'sk', ...); discarding the audio would lose
            # perfectly valid speech.
            if detected_language not in SUPPORTED_LANGUAGES:
                fallback = last_confident_language or DEFAULT_LANGUAGE
                logger.warning(
                    f"[TRANSCRIBER] Unsupported language '{detected_language}' "
                    f"(p={language_probability:.2f}) — falling back to '{fallback}'"
                )
                detected_language = fallback

            # Low confidence on a final: still transcribe. Discarding the whole
            # utterance over uncertain language detection loses valid audio; the
            # translator's same-language guard prevents inverted translations.
            elif is_final and language_probability < LANGUAGE_CONFIDENCE_THRESHOLD:
                fallback = last_confident_language or detected_language
                logger.warning(
                    f"[TRANSCRIBER] Low language confidence ({detected_language}, "
                    f"p={language_probability:.2f}) — proceeding with '{fallback}'"
                )
                detected_language = fallback

            # Guard clause: empty audio already handled above. Build the
            # transcript, dropping any words/segments whose timestamps fall
            # inside the overlap window (Step 5) so the previous segment's
            # tail is never duplicated in the output.
            text = _join_transcript(segments, overlap_seconds)

            transcription_end = time.time()
            transcription_elapsed = transcription_end - transcription_start
            
            # Guard clause: No text produced. This is the genuine silence case
            # (VAD removed everything — e.g. Stereo Mix received no audio), and
            # the only path that cancels a final.
            if not text:
                if is_final:
                    _send_to_queue(ui_queue, ui_cancel("no_speech"))
                    logger.warning("[TRANSCRIBER] No speech detected in final chunk — sent cancel")
                continue

            # Valid text branch
            if not is_final:
                # Accumulate the growing transcript (each partial covers the
                # audio slice since the previous one).
                provisional_text = (provisional_text + " " + text).strip()
                _send_to_queue(ui_queue, ui_partial(provisional_text))
                # Progressive translation: forward the accumulated text as a
                # provisional task so the LLM translates while the user is
                # still speaking. Throttled by text growth to limit LLM load.
                if len(provisional_text) - last_provisional_sent_len >= PARTIAL_PROGRESS_THRESHOLD:
                    _send_to_queue(
                        translation_queue, (provisional_text, detected_language, {}, True),
                        block=False, error_msg="translation_queue full, dropped provisional"
                    )
                    last_provisional_sent_len = len(provisional_text)
            else:
                provisional_text = ""
                last_provisional_sent_len = 0
                logger.info(
                    f"[TRANSCRIBER] Transcription completed in {transcription_elapsed:.3f}s "
                    f"({len(text)} chars): '{text}'"
                )
                # Chain this final's text as context for the next chunk so
                # Whisper continues the sentence naturally (Phase 8.2).
                last_final_text = text

                # Propagate timing dict with transcription timestamps
                timing["transcription_start"] = transcription_start
                timing["transcription_end"] = transcription_end

                # Terminal event contract: every final path emits at least one
                # terminal UI event. If a final cannot be delivered, emit a
                # 'skipped' event so the UI never locks in a pending state.
                final_sent = _send_to_queue(ui_queue, ui_final(text), block=True, timeout=TIMEOUT_PUT, error_msg="ui_queue full, dropped final transcription")
                if not final_sent:
                    _send_to_queue(ui_queue, ui_skipped("queue_full", stage="ui"), block=False)

                translation_sent = _send_to_queue(translation_queue, (text, detected_language, timing), block=True, timeout=TIMEOUT_PUT, error_msg="translation_queue full, dropped text")
                if not translation_sent:
                    _send_to_queue(ui_queue, ui_skipped("queue_full", stage="translation"), block=True, timeout=TIMEOUT_PUT)
                else:
                    logger.info(f"[TRANSCRIBER] Final queued for translation (lang={detected_language}, {len(text)} chars)")

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Transcriber error: {e}. Item: {item!r}")
            _send_to_queue(ui_queue, ui_error(f"Transcription Error: {e}"), block=True, timeout=TIMEOUT_PUT)
