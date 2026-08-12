import multiprocessing
import queue
import logging
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

# Constants
MODEL_EN_ES = "Helsinki-NLP/opus-mt-en-es"
MODEL_ES_EN = "Helsinki-NLP/opus-mt-es-en"
MAX_LENGTH = 512
TIMEOUT_GET = 1.0
TIMEOUT_PUT = 5.0

def _send_to_queue(q, msg, block=False, timeout=None, error_msg="Queue put failed"):
    try:
        q.put(msg, block=block, timeout=timeout)
    except queue.Full:
        if block:
            logger.error(error_msg)
    except Exception as e:
        logger.debug(f"Queue communication error: {e}")

def start_translator(
    translation_queue: multiprocessing.Queue, ui_queue: multiprocessing.Queue
):
    """
    Translates text based on detected language using MarianMT.
    Pulls (text, lang) tuples from translation_queue and pushes
    (original_text, translated_text) tuples to ui_queue.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    en_to_es = pipeline("translation", model=MODEL_EN_ES, device=device, max_length=MAX_LENGTH, truncation=True)
    es_to_en = pipeline("translation", model=MODEL_ES_EN, device=device, max_length=MAX_LENGTH, truncation=True)

    _send_to_queue(ui_queue, {"type": "status", "process": "translator", "status": "ready"})

    while True:
        try:
            item = translation_queue.get(timeout=TIMEOUT_GET)
            if item is None:
                break
                
            if not isinstance(item, tuple) or len(item) != 2:
                continue

            text, lang = item
            if not text:
                continue

            translated_text = text
            if lang == "en":
                result = en_to_es(text)
                translated_text = result[0]['translation_text']
            elif lang == "es":
                result = es_to_en(text)
                translated_text = result[0]['translation_text']
                
            _send_to_queue(
                ui_queue, 
                {"type": "translation", "original": text, "translated": translated_text}, 
                block=True, 
                timeout=TIMEOUT_PUT, 
                error_msg="ui_queue full, dropped translation"
            )
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Translator error: {e}")
            _send_to_queue(ui_queue, {"type": "cancel"})
