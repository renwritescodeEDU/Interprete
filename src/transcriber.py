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
    model = WhisperModel("small", device="auto", compute_type="int8")
    
    # Notify UI that transcriber is ready
    try:
        ui_queue.put({"type": "status", "process": "transcriber", "status": "ready"}, block=False)
    except queue.Full:
        pass

    detected_language = None

    while True:
        try:
            item = asr_queue.get(timeout=1)
            if item is None:
                break
            
            if item == "QUIT":
                break

            if not isinstance(item, tuple) or len(item) != 3:
                continue

            audio_data, rate, is_final = item
            if audio_data is not None and len(audio_data) > 0:
                if detected_language == "es":
                    prompt = "Hola. Esta es una transcripción en español perfecta, con excelente ortografía, puntuación y gramática."
                elif detected_language == "en":
                    prompt = "Hello. This is a perfect English transcription, with excellent spelling, punctuation, and grammar."
                else:
                    prompt = None
                    
                segments, info = model.transcribe(
                    audio_data, 
                    beam_size=5, 
                    vad_filter=True,
                    initial_prompt=prompt
                )
                
                detected_language = info.language
                
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
                            detected_language = None
                            # Send final status to UI
                            try:
                                ui_queue.put({"type": "final", "text": text}, block=True, timeout=5.0)
                            except queue.Full:
                                print("Error: ui_queue full, dropped final transcription")
                            
                            # Send to translator
                            try:
                                translation_queue.put((text, info.language), block=True, timeout=5.0)
                            except queue.Full:
                                print("Error: translation_queue full, dropped text")
                    else:
                        if is_final:
                            detected_language = None
                            try: ui_queue.put({"type": "cancel"}, block=False)
                            except: pass
                else:
                    if is_final:
                        detected_language = None
                        try: ui_queue.put({"type": "cancel"}, block=False)
                        except: pass
            else:
                if is_final:
                    detected_language = None
                    try: ui_queue.put({"type": "cancel"}, block=False)
                    except: pass
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Transcriber error: {e}")
            try:
                ui_queue.put({"type": "error", "message": f"Transcription Error: {e}"}, block=True, timeout=5.0)
            except Exception:
                pass
