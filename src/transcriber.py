import multiprocessing
import queue
import time
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Constants
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
SUPPORTED_LANGUAGES = {"en", "es"}
TIMEOUT_PUT = 5.0
TIMEOUT_GET = 1.0
# Minimum probability for language detection to be trusted.
# Fragments below this threshold are discarded to avoid inverted translations.
LANGUAGE_CONFIDENCE_THRESHOLD = 0.60

PROMPT_ES = "Hola. Esta es una transcripción en español perfecta, con excelente ortografía, puntuación y gramática."
PROMPT_EN = "Hello. This is a perfect English transcription, with excellent spelling, punctuation, and grammar."

def _send_to_queue(q, msg, block=False, timeout=None, error_msg="Queue put failed"):
    try:
        q.put(msg, block=block, timeout=timeout)
    except queue.Full:
        if block:
            logger.error(error_msg)
    except Exception as e:
        logger.debug(f"Queue communication error: {e}")

def start_transcriber(
    asr_queue: multiprocessing.Queue, translation_queue: multiprocessing.Queue, ui_queue: multiprocessing.Queue
):
    """
    Pulls audio chunks from asr_queue, transcribes them, and pushes (text, language, timing) tuples
    to translation_queue. Supports 3-element (legacy) and 4-element (with timing) tuples from audio.
    """
    model = WhisperModel(MODEL_SIZE, device="auto", compute_type=COMPUTE_TYPE)
    
    _send_to_queue(ui_queue, {"type": "status", "process": "transcriber", "status": "ready"})

    detected_language = None

    while True:
        try:
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
            
            # Guard clause: Empty audio
            if audio_data is None or len(audio_data) == 0:
                if is_final:
                    detected_language = None
                    _send_to_queue(ui_queue, {"type": "cancel"})
                continue

            transcription_start = time.time()

            # Determine initial prompt based on previously detected language
            prompt = None
            if detected_language == "es":
                prompt = PROMPT_ES
            elif detected_language == "en":
                prompt = PROMPT_EN
                
            segments, info = model.transcribe(
                audio_data, 
                beam_size=BEAM_SIZE, 
                vad_filter=True,
                initial_prompt=prompt
            )
            
            detected_language = info.language
            language_probability = info.language_probability
            
            # Guard clause: Unsupported language
            if detected_language not in SUPPORTED_LANGUAGES:
                if is_final:
                    detected_language = None
                    _send_to_queue(ui_queue, {"type": "cancel"})
                continue

            # Guard clause: Low confidence — drop to avoid inverted translations
            if is_final and language_probability < LANGUAGE_CONFIDENCE_THRESHOLD:
                logger.warning(
                    f"[TRANSCRIBER] Discarding low-confidence fragment "
                    f"(lang='{detected_language}', prob={language_probability:.2f} < {LANGUAGE_CONFIDENCE_THRESHOLD})"
                )
                detected_language = None
                _send_to_queue(ui_queue, {"type": "cancel"})
                continue

            text = "".join(segment.text for segment in segments).strip()

            transcription_end = time.time()
            transcription_elapsed = transcription_end - transcription_start
            
            # Guard clause: No text produced
            if not text:
                if is_final:
                    detected_language = None
                    _send_to_queue(ui_queue, {"type": "cancel"})
                continue

            # Valid text branch
            if not is_final:
                _send_to_queue(ui_queue, {"type": "partial", "text": text})
            else:
                detected_language = None
                logger.info(f"[TRANSCRIBER] Transcription completed in {transcription_elapsed:.3f}s: '{text[:80]}{'...' if len(text) > 80 else ''}'")

                # Propagate timing dict with transcription timestamps
                timing["transcription_start"] = transcription_start
                timing["transcription_end"] = transcription_end

                _send_to_queue(ui_queue, {"type": "final", "text": text}, block=True, timeout=TIMEOUT_PUT, error_msg="ui_queue full, dropped final transcription")
                _send_to_queue(translation_queue, (text, info.language, timing), block=True, timeout=TIMEOUT_PUT, error_msg="translation_queue full, dropped text")

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Transcriber error: {e}")
            _send_to_queue(ui_queue, {"type": "error", "message": f"Transcription Error: {e}"}, block=True, timeout=TIMEOUT_PUT)
