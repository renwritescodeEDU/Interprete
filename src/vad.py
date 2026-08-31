"""Voice activity detection with automatic RMS fallback.

``VoiceDetector`` tries to load the Silero VAD ONNX model at initialisation.
If the model file is not found or ``onnxruntime`` is unavailable, it sets
``using_rms_fallback = True`` and classifies frames with a per-frame RMS
energy threshold — the exact behaviour of the pre-VAD capture loop.

Model resolution order (first hit wins):

1. Explicit ``model_path`` argument.
2. ``INTERPRETE_VAD_MODEL`` environment variable.
3. ``<repo_root>/models/silero_vad_v6.onnx``.

The model is opt-in on purpose: the existing auto-commit tests use
constant-amplitude synthetic frames that Silero VAD correctly classifies
as non-speech (they are not speech-like signals), so an unconditional
auto-discovery of the faster-whisper bundled model would break the
Zero-Breakage guarantee. Copy ``silero_vad_v6.onnx`` into ``models/``
(or point ``INTERPRETE_VAD_MODEL`` at it) to enable real VAD.
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

RMS_THRESHOLD = 0.005
VAD_THRESHOLD = 0.5

_SILERO_NUM_SAMPLES = 512
_SILERO_CONTEXT = 64

try:
    import onnxruntime
    _ONNX_AVAILABLE = True
except ImportError:
    onnxruntime = None
    _ONNX_AVAILABLE = False


def _resolve_model_path(model_path: str | None = None) -> str | None:
    """Return a usable Silero VAD model path, or None to use RMS fallback."""
    candidates = []
    if model_path:
        candidates.append(model_path)
    env_path = os.environ.get("INTERPRETE_VAD_MODEL")
    if env_path:
        candidates.append(env_path)
    candidates.append(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "silero_vad_v6.onnx")
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


class VoiceDetector:
    """Silero VAD frame classifier with RMS fallback.

    The capture loop calls :meth:`is_speech` once per 30 ms frame. In VAD
    mode the classifier accumulates samples into a 512-sample window (the
    Silero model's native window); once a full window is available the
    ONNX model runs and the returned probability decides speech/silence.
    """

    def __init__(
        self,
        model_path: str | None = None,
        vad_threshold: float = VAD_THRESHOLD,
        rms_threshold: float = RMS_THRESHOLD,
    ):
        self.vad_threshold = vad_threshold
        self.rms_threshold = rms_threshold
        self.using_rms_fallback = True
        self._session = None
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._vad_window = np.zeros(0, dtype=np.float32)

        resolved = _resolve_model_path(model_path)
        if resolved is None:
            logger.info(
                "[VAD] No Silero VAD model found (models/silero_vad_v6.onnx "
                "or INTERPRETE_VAD_MODEL); using RMS fallback"
            )
            return

        if onnxruntime is None:
            logger.info("[VAD] onnxruntime not available; using RMS fallback")
            return

        try:
            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.enable_cpu_mem_arena = False
            opts.log_severity_level = 4

            self._session = onnxruntime.InferenceSession(
                resolved,
                providers=["CPUExecutionProvider"],
                sess_options=opts,
            )
            self.using_rms_fallback = False
            logger.info("[VAD] Silero VAD loaded from %s", resolved)
        except Exception as exc:
            logger.warning("[VAD] Failed to load Silero VAD (%s); using RMS fallback", exc)

    def reset(self) -> None:
        """Clear the sample window and the recurrent state.

        Called when a new recording segment starts so the classifier does
        not carry context from a previous conversation.
        """
        self._vad_window = np.zeros(0, dtype=np.float32)
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)

    def is_speech(self, frame_bytes: bytes) -> bool:
        if self.using_rms_fallback:
            return self._rms_is_speech(frame_bytes)
        return self._vad_is_speech(frame_bytes)

    def _rms_is_speech(self, frame_bytes: bytes) -> bool:
        frame = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(frame))))
        return rms >= self.rms_threshold

    def _vad_is_speech(self, frame_bytes: bytes) -> bool:
        frame = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._vad_window = np.concatenate([self._vad_window, frame])
        if len(self._vad_window) < _SILERO_NUM_SAMPLES:
            return False
        window = self._vad_window[:_SILERO_NUM_SAMPLES]
        self._vad_window = self._vad_window[_SILERO_NUM_SAMPLES:]
        prob = self._run_silero(window)
        return prob >= self.vad_threshold

    def _run_silero(self, audio: np.ndarray) -> float:
        """Run the Silero v6 model on a 512-sample window.

        Replicates the faster-whisper protocol: each 512-sample chunk is
        prefixed with a 64-sample context from the previous chunk and fed
        to the recurrent LSTM (h/c carry state between calls).
        """
        batched = audio.reshape(-1, _SILERO_NUM_SAMPLES)
        ctx = batched[..., -_SILERO_CONTEXT:]
        ctx[-1] = 0
        ctx = np.roll(ctx, 1, 0)
        batched = np.concatenate([ctx, batched], 1)
        batched = batched.reshape(-1, _SILERO_NUM_SAMPLES + _SILERO_CONTEXT)

        out, self._h, self._c = self._session.run(
            None,
            {"input": batched, "h": self._h, "c": self._c},
        )
        return float(np.asarray(out).reshape(-1)[0])