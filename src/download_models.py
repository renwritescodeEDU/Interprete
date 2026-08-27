import sys
import subprocess
import logging

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:1.5b"

def download_models():
    """Download required local LLM models using Ollama CLI."""
    logger.info(f"Pulling LLM model '{LLM_MODEL}' via Ollama...")
    try:
        subprocess.run(["ollama", "pull", LLM_MODEL], check=True)
        logger.info(f"Model '{LLM_MODEL}' is ready.")
    except Exception as e:
        logger.error(f"Failed to pull '{LLM_MODEL}'. Is Ollama installed and running? Error: {e}")
        sys.exit(1)
        
    logger.info("All dependencies downloaded successfully.")

if __name__ == "__main__":
    download_models()
