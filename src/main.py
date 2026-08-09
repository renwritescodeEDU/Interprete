import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing
from datetime import datetime
from src.audio import start_audio_capture
from src.transcriber import start_transcriber
from src.translator import start_translator
from src.ui import run_ui


class Orchestrator:
    def __init__(self):
        self.audio_process = None
        self.asr_process = None
        self.translator_process = None
        
        # Protect against queue desync lag
        self.asr_queue = multiprocessing.Queue(maxsize=5)
        self.translation_queue = multiprocessing.Queue(maxsize=5)
        self.ui_queue = multiprocessing.Queue()
        
        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join("logs", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def start_processes(self):
        print("Starting background processes...")
        # Clear queues if restarting
        for q in [self.asr_queue, self.translation_queue, self.ui_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    pass
        
        self.audio_process = multiprocessing.Process(
            target=start_audio_capture, args=(self.asr_queue,), daemon=True
        )
        self.asr_process = multiprocessing.Process(
            target=start_transcriber, args=(self.asr_queue, self.translation_queue), daemon=True
        )
        self.translator_process = multiprocessing.Process(
            target=start_translator, args=(self.translation_queue, self.ui_queue), daemon=True
        )

        self.audio_process.start()
        self.asr_process.start()
        self.translator_process.start()
        print("Background processes running.")

    def stop_processes(self):
        print("Stopping background processes...")
        # Send poison pills
        self.asr_queue.put(None)
        self.translation_queue.put(None)
        
        if self.audio_process and self.audio_process.is_alive():
            self.audio_process.terminate()
            self.audio_process.join(timeout=1)
            
        if self.asr_process and self.asr_process.is_alive():
            self.asr_process.join(timeout=2)
            if self.asr_process.is_alive():
                self.asr_process.terminate()
                
        if self.translator_process and self.translator_process.is_alive():
            self.translator_process.join(timeout=2)
            if self.translator_process.is_alive():
                self.translator_process.terminate()
        
        print("Background processes stopped.")


def main():
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    orchestrator = Orchestrator()
    run_ui(orchestrator.ui_queue, orchestrator.start_processes, orchestrator.stop_processes, orchestrator.log_path)


if __name__ == "__main__":
    main()
