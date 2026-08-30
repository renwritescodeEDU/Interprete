"""
Integration tests for the pipeline's latency budget.

These tests are marked 'slow' and are NOT run by default (use
`pytest -m slow`). They require the real stack to be available:

- faster-whisper model "small" downloaded
- Ollama running with the llama3.2:3b model
- an audio input device (microphone or BlackHole virtual device)

They exist to validate the ~2s stop->display budget on target hardware
and to catch regressions in the beam-size tuning.
"""

import os
import sys
import time
import queue
import multiprocessing

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio import start_audio_capture
from src.transcriber import start_transcriber, BEAM_SIZE, BEAM_SIZE_PARTIAL
from src.translator import start_translator

# Budget from stop-of-recording to translation displayed in the UI.
LATENCY_BUDGET_STOP_TO_DISPLAY = 2.0


class TestPipelineLatency:
    """Validates the live-interpretation latency budget."""

    @pytest.fixture()
    def pipeline_queues(self):
        return (
            multiprocessing.Queue(),
            multiprocessing.Queue(),
            multiprocessing.Queue(),
        )

    def test_beam_size_contract(self):
        """Partials must use a smaller beam than finals (latency vs quality)."""
        assert BEAM_SIZE_PARTIAL < BEAM_SIZE, (
            f"Partial beam ({BEAM_SIZE_PARTIAL}) should be smaller than "
            f"final beam ({BEAM_SIZE})"
        )

    @pytest.mark.slow
    def test_stop_to_display_latency(self, pipeline_queues):
        """
        Records a short utterance through the audio device, runs it through
        the full ASR->translation pipeline, and asserts the wall-clock time
        from recording-stop to translation display is within budget.

        Requires real hardware/models. Run with: pytest -m slow
        """
        pytest.skip(
            "Requires a microphone, faster-whisper 'small', and Ollama. "
            "Run manually on target hardware."
        )

        asr_queue, translation_queue, ui_queue = pipeline_queues
        control_queue = multiprocessing.Queue()

        processes = [
            multiprocessing.Process(
                target=start_audio_capture,
                # Keyword args keep this aligned with the signature if the
                # worker gains more optional parameters (device_index etc.).
                kwargs={
                    "asr_queue": asr_queue,
                    "control_queue": control_queue,
                    "ui_queue": ui_queue,
                },
                daemon=True,
            ),
            multiprocessing.Process(
                target=start_transcriber,
                args=(asr_queue, translation_queue, ui_queue),
                daemon=True,
            ),
            multiprocessing.Process(
                target=start_translator,
                args=(translation_queue, ui_queue),
                daemon=True,
            ),
        ]
        for p in processes:
            p.start()

        try:
            start_ts = time.time()
            control_queue.put(("START", start_ts))

            # Recording window. Default 3s; override with INTERPRETE_RECORD_SECONDS
            # to give a human speaker time to start speaking. The stop->display
            # measurement begins at FINISH, so a longer window is measurement-safe.
            record_seconds = float(os.environ.get("INTERPRETE_RECORD_SECONDS", "3.0"))
            time.sleep(record_seconds)
            stop_ts = time.time()
            control_queue.put(("FINISH", stop_ts))

            translation_time = None
            deadline = time.time() + LATENCY_BUDGET_STOP_TO_DISPLAY + 5.0
            while time.time() < deadline:
                try:
                    msg = ui_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if msg.get("type") == "translation":
                    translation_time = time.time()
                    break

            assert translation_time is not None, "No translation reached the UI"
            latency = translation_time - stop_ts
            assert latency <= LATENCY_BUDGET_STOP_TO_DISPLAY, (
                f"stop->display latency {latency:.2f}s exceeds "
                f"{LATENCY_BUDGET_STOP_TO_DISPLAY:.1f}s budget"
            )
        finally:
            for q in (control_queue, asr_queue, translation_queue):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            for p in processes:
                p.join(timeout=2.0)
                if p.is_alive():
                    p.terminate()
