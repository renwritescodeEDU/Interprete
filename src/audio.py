import multiprocessing
import queue
import numpy as np
import pyaudio
import webrtcvad

CHUNK = 480  # 30ms at 16000Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def start_audio_capture(asr_queue: multiprocessing.Queue):
    """
    Captures audio from default input, chunks it using VAD,
    and pushes tuples of (audio_array, sample_rate) to asr_queue.
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
    silence_frames = 0
    max_frames = int((15.0 * RATE) / CHUNK)  # 15 seconds
    silence_threshold = int((1.5 * RATE) / CHUNK)  # 1500ms
    min_frames = int((0.5 * RATE) / CHUNK)  # 0.5s minimum to discard noise

    try:
        while True:
            # exception_on_overflow=False prevents crashes on slow processing
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)

            if is_speech:
                silence_frames = 0
                frames.append(frame)
            else:
                if len(frames) > 0:
                    silence_frames += 1
                    frames.append(frame)

            # Flush when silence threshold is met or max duration reached
            if len(frames) > 0:
                if silence_frames >= silence_threshold or len(frames) >= max_frames:
                    if len(frames) > min_frames:
                        audio_bytes = b"".join(frames)
                        # Convert 16-bit PCM to float32
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        try:
                            asr_queue.put((audio_array, RATE), block=False)
                        except queue.Full:
                            pass
                    
                    frames = []
                    silence_frames = 0
    except Exception as e:
        print(f"Audio capture terminated: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
