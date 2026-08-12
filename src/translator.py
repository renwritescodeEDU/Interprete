import multiprocessing
import queue
import torch
from transformers import pipeline


def start_translator(
    translation_queue: multiprocessing.Queue, ui_queue: multiprocessing.Queue
):
    """
    Translates text based on detected language using MarianMT.
    Pulls (text, lang) tuples from translation_queue and pushes
    (original_text, translated_text) tuples to ui_queue.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    en_to_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=device, max_length=512, truncation=True)
    es_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=device, max_length=512, truncation=True)

    # Notify UI that translator is ready
    try:
        ui_queue.put({"type": "status", "process": "translator", "status": "ready"}, block=False)
    except queue.Full:
        pass

    while True:
        try:
            item = translation_queue.get(timeout=1)
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
                
            try:
                ui_queue.put({
                    "type": "translation",
                    "original": text,
                    "translated": translated_text
                }, block=True, timeout=5.0)
            except queue.Full:
                print("Error: ui_queue full, dropped translation")
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Translator error: {e}")
            try:
                ui_queue.put({"type": "cancel"}, block=False)
            except:
                pass
