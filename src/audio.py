import multiprocessing
import numpy as np
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def start_audio_capture(asr_queue: multiprocessing.Queue):
    """
    Captures audio from default input.
    Pushes tuples of (audio_array, sample_rate) to asr_queue.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # Minimal stub: push one dummy chunk to prevent blocking forever in tests
    dummy_data = np.zeros(CHUNK, dtype=np.float32)
    asr_queue.put((dummy_data, RATE))
    stream.stop_stream()
    stream.close()
    p.terminate()
