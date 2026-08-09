import multiprocessing
import queue
from faster_whisper import WhisperModel


def start_transcriber(
    asr_queue: multiprocessing.Queue, translation_queue: multiprocessing.Queue, ui_queue: multiprocessing.Queue
):
    """
    Pulls audio chunks from asr_queue, transcribes them, and pushes (text, language) tuples
    to translation_queue.
    """
    model = WhisperModel("tiny", device="auto", compute_type="int8")

    while True:
        try:
            item = asr_queue.get(timeout=1)
            if item is None:
                break

            if not isinstance(item, tuple) or len(item) != 3:
                continue

            audio_data, rate, is_final = item
            if audio_data is not None and len(audio_data) > 0:
                segments, info = model.transcribe(audio_data, beam_size=1)
                
                if info.language in ["en", "es"]:
                    text = "".join(segment.text for segment in segments).strip()
                    if text:
                        if not is_final:
                            # Send partial to UI
                            try:
                                ui_queue.put({"type": "partial", "text": text}, block=False)
                            except queue.Full:
                                pass
                        else:
                            # Send to translator
                            try:
                                translation_queue.put((text, info.language), block=False)
                            except queue.Full:
                                pass
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Transcriber error: {e}")
