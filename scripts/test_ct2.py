from faster_whisper import WhisperModel
import time
print("Loading model with auto device...")
try:
    model = WhisperModel("tiny", device="auto", compute_type="int8")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed: {e}")
