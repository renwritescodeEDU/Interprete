import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing
import logging
from datetime import datetime
from src.audio import start_audio_capture
from src.transcriber import start_transcriber
from src.translator import start_translator
from src.ui import run_ui

# Constants
MAX_QUEUE_SIZE = 5
PROCESS_JOIN_TIMEOUT = 3.0
PROCESS_TERM_TIMEOUT = 1.0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self.audio_process = None
        self.asr_process = None
        self.translator_process = None
        
        # Protect against queue desync lag
        self.asr_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
        self.translation_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
        self.ui_queue = multiprocessing.Queue()
        self.control_queue = multiprocessing.Queue()
        
        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join("logs", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def _clear_queue(self, q: multiprocessing.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                break

    def start_processes(self):
        logger.info("Starting background processes...")
        # Clear queues if restarting
        for q in [self.asr_queue, self.translation_queue, self.ui_queue, self.control_queue]:
            self._clear_queue(q)
        
        self.audio_process = multiprocessing.Process(
            target=start_audio_capture, args=(self.asr_queue, self.control_queue, self.ui_queue, None), daemon=True
        )
        self.asr_process = multiprocessing.Process(
            target=start_transcriber, args=(self.asr_queue, self.translation_queue, self.ui_queue), daemon=True
        )
        self.translator_process = multiprocessing.Process(
            target=start_translator, args=(self.translation_queue, self.ui_queue), daemon=True
        )

        self.audio_process.start()
        self.asr_process.start()
        self.translator_process.start()
        logger.info("Background processes running.")

    def _terminate_process(self, process, name):
        if process and process.is_alive():
            process.join(timeout=PROCESS_JOIN_TIMEOUT)
            if process.is_alive():
                logger.warning(f"Process {name} did not join, forcing terminate.")
                process.terminate()
                process.join(timeout=PROCESS_TERM_TIMEOUT)

    def stop_processes(self):
        logger.info("Stopping background processes...")
        
        for q in [self.control_queue, self.asr_queue, self.translation_queue]:
            self._clear_queue(q)
                    
        # Send poison pills
        for q, msg in [(self.control_queue, "QUIT"), (self.asr_queue, None), (self.translation_queue, None)]:
            try:
                q.put_nowait(msg)
            except Exception as e:
                logger.debug(f"Failed to send poison pill: {e}")
        
        self._terminate_process(self.audio_process, "audio")
        self._terminate_process(self.asr_process, "transcriber")
        self._terminate_process(self.translator_process, "translator")
        
        logger.info("Background processes stopped.")


def main():
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    orchestrator = Orchestrator()
    run_ui(
        orchestrator.ui_queue, 
        orchestrator.control_queue, 
        orchestrator.start_processes, 
        orchestrator.stop_processes, 
        orchestrator.log_path
    )

if __name__ == "__main__":
    main()
