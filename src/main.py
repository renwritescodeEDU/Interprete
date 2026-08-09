import multiprocessing
from src.audio import start_audio_capture
from src.transcriber import start_transcriber
from src.translator import start_translator
from src.ui import run_ui


def main():
    asr_queue = multiprocessing.Queue()
    translation_queue = multiprocessing.Queue()
    ui_queue = multiprocessing.Queue()

    # In production, these are multiprocessing.Process(target=...)
    # For now, just setup the structure.
    print("Orchestrator ready.")


if __name__ == "__main__":
    main()
