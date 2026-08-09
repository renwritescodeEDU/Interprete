import multiprocessing
import signal
from src.audio import start_audio_capture
from src.transcriber import start_transcriber
from src.translator import start_translator
from src.ui import run_ui


def main():
    # macOS requires 'spawn' to avoid fork-safety issues with CoreFoundation/AppKit and PyTorch.
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    asr_queue = multiprocessing.Queue()
    translation_queue = multiprocessing.Queue()
    ui_queue = multiprocessing.Queue()

    audio_process = multiprocessing.Process(
        target=start_audio_capture, args=(asr_queue,), daemon=True
    )
    asr_process = multiprocessing.Process(
        target=start_transcriber, args=(asr_queue, translation_queue), daemon=True
    )
    translator_process = multiprocessing.Process(
        target=start_translator, args=(translation_queue, ui_queue), daemon=True
    )

    print("Starting background processes...")
    audio_process.start()
    asr_process.start()
    translator_process.start()

    print("Background processes running. Launching UI. Press Ctrl+C in terminal to exit.")
    # Intercept SIGINT so PyQt handles Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Launch GUI on the main thread
    run_ui(ui_queue)


if __name__ == "__main__":
    main()
