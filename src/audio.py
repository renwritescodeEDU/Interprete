import multiprocessing
import queue
import numpy as np
import pyaudio
import webrtcvad

CHUNK = 480  # 30ms at 16000Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def start_audio_capture(asr_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue):
    """
    Captures audio from default input, chunks it using VAD,
    and pushes tuples of (audio_array, sample_rate, is_final) to asr_queue.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    vad = webrtcvad.Vad(3)
    frames = []
    
    # Send partial transcript every ~1 second of accumulated speech
    partial_threshold = int((1.0 * RATE) / CHUNK)  
    frames_since_last_partial = 0
    is_listening = False

    try:
        while True:
            # Check control queue for FINISH command
            try:
                cmd = control_queue.get_nowait()
                if cmd == "FINISH":
                    if len(frames) > 0:
                        audio_bytes = b"".join(frames)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        try:
                            asr_queue.put((audio_array, RATE, True), block=False)
                        except queue.Full:
                            pass
                    frames = []
                    frames_since_last_partial = 0
                    is_listening = False
            except queue.Empty:
                pass

            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)

            if is_speech or is_listening:
                frames.append(frame)
                frames_since_last_partial += 1
                is_listening = True
                
                # Emit a partial transcript periodically
                if frames_since_last_partial >= partial_threshold:
                    audio_bytes = b"".join(frames)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    try:
                        asr_queue.put((audio_array, RATE, False), block=False)
                    except queue.Full:
                        pass
                    frames_since_last_partial = 0
    except Exception as e:
        print(f"Audio capture terminated: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
