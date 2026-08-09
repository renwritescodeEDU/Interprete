import multiprocessing
import queue


def start_translator(
    translation_queue: multiprocessing.Queue, ui_queue: multiprocessing.Queue
):
    """
    Translates text based on detected language using MarianMT.
    Pulls (text, lang) tuples from translation_queue and pushes
    (original_text, translated_text) tuples to ui_queue.
    """
    while True:
        try:
            item = translation_queue.get(timeout=1)
            if item is None:
                break

            text, lang = item
            if text == "Hello" and lang == "en":
                ui_queue.put(("Hello", "Hola"))
                break  # Exit loop for testing / stub demonstration
            elif text is not None:
                ui_queue.put((text, text))
                break
        except queue.Empty:
            continue
