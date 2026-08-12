from transformers import pipeline
from faster_whisper import WhisperModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants matching runtime files
MODEL_EN_ES = "Helsinki-NLP/opus-mt-en-es"
MODEL_ES_EN = "Helsinki-NLP/opus-mt-es-en"
WHISPER_MODEL = "small"

def download_models():
    logger.info(f"Downloading English -> Spanish model ({MODEL_EN_ES})...")
    pipeline("translation", model=MODEL_EN_ES)
    
    logger.info(f"Downloading Spanish -> English model ({MODEL_ES_EN})...")
    pipeline("translation", model=MODEL_ES_EN)
    
    logger.info(f"Downloading Whisper ASR model ({WHISPER_MODEL})...")
    WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    
    logger.info("Models downloaded and cached successfully. The system is ready for offline use.")

if __name__ == "__main__":
    download_models()
