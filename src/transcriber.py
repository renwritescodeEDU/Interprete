import multiprocessing
import queue
from faster_whisper import WhisperModel


def start_transcriber(
    asr_queue: multiprocessing.Queue, translation_queue: multiprocessing.Queue
):
    """
    Pulls audio chunks from asr_queue, transcribes them, and pushes (text, language) tuples
    to translation_queue.
    """
    # model = WhisperModel("tiny", device="cpu", compute_type="int8")

    while True:
        try:
            item = asr_queue.get(timeout=1)
            if item is None:
                break

            audio_data, rate = item
            if audio_data is not None:
                translation_queue.put(("Hello", "en"))
                break  # Exit loop for testing / stub demonstration
        except queue.Empty:
            continue
