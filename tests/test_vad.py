"""Tests for the VoiceDetector (Silero VAD + RMS fallback).

Covered:

- VAD mode classifies speech/silence from the ONNX model probability
  (the session is mocked; the real model is never loaded here).
- ``onnxruntime`` unavailable -> RMS fallback, no exception.
- Model file missing -> RMS fallback, no exception.
- The RMS fallback reproduces the original energy-threshold behaviour.
- ``reset()`` clears the sample window and the recurrent state.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.vad import VoiceDetector

FRAME_SPEECH = b'\x10\x27' * 480
FRAME_SILENCE = b'\x00\x00' * 480


class TestVoiceDetectorFallback(unittest.TestCase):
    """RMS fallback paths: model missing or onnxruntime unavailable."""

    def test_fallback_when_model_missing(self):
        """A nonexistent model path must not raise and must use RMS."""
        detector = VoiceDetector(model_path="/definitely/not/here/model.onnx")
        self.assertTrue(detector.using_rms_fallback)
        # Loud constant frame: RMS = 10000/32768 ≈ 0.305 ≥ 0.005 -> speech
        self.assertTrue(detector.is_speech(FRAME_SPEECH))
        # Silence frame: RMS = 0.0 < 0.005 -> not speech
        self.assertFalse(detector.is_speech(FRAME_SILENCE))

    @patch("src.vad.onnxruntime", None)
    def test_fallback_when_onnxruntime_unavailable(self):
        """Simulate onnxruntime not installed: must fall back to RMS."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            fake_model = f.name
        try:
            detector = VoiceDetector(model_path=fake_model)
            self.assertTrue(detector.using_rms_fallback)
            self.assertTrue(detector.is_speech(FRAME_SPEECH))
            self.assertFalse(detector.is_speech(FRAME_SILENCE))
        finally:
            os.unlink(fake_model)

    @patch("src.vad.onnxruntime")
    def test_fallback_when_session_init_fails(self, mock_ort):
        """InferenceSession raising must degrade to RMS, never crash."""
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("ONNX parse error")
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            fake_model = f.name
        try:
            detector = VoiceDetector(model_path=fake_model)
            self.assertTrue(detector.using_rms_fallback)
            self.assertTrue(detector.is_speech(FRAME_SPEECH))
            self.assertFalse(detector.is_speech(FRAME_SILENCE))
        finally:
            os.unlink(fake_model)


class TestVoiceDetectorVADMode(unittest.TestCase):
    """VAD-mode classification with a mocked ONNX session."""

    def _detector_with_probability(self, probability):
        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_session.run.return_value = (
            np.array([[probability]], dtype=np.float32),
            np.zeros((1, 1, 128), dtype=np.float32),
            np.zeros((1, 1, 128), dtype=np.float32),
        )
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        temp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        temp.close()
        self.addCleanup(os.unlink, temp.name)
        with patch("src.vad.onnxruntime", mock_ort):
            detector = VoiceDetector(model_path=temp.name)
        return detector

    def test_vad_classifies_speech(self):
        """Probability 0.85 (>= 0.5) must classify the frame as speech."""
        detector = self._detector_with_probability(0.85)
        self.assertFalse(detector.using_rms_fallback)
        # First call only accumulates samples (480 < 512 window) -> no decision.
        self.assertFalse(detector.is_speech(FRAME_SPEECH))
        # Second call fills the window and returns the model decision.
        self.assertTrue(detector.is_speech(FRAME_SPEECH))

    def test_vad_classifies_silence(self):
        """Probability 0.05 (< 0.5) must classify the frame as silence."""
        detector = self._detector_with_probability(0.05)
        self.assertFalse(detector.using_rms_fallback)
        self.assertFalse(detector.is_speech(FRAME_SPEECH))
        self.assertFalse(detector.is_speech(FRAME_SPEECH))

    def test_vad_threshold_boundary(self):
        """Probability exactly at the threshold counts as speech (>=)."""
        detector = self._detector_with_probability(0.5)
        self.assertFalse(detector.is_speech(FRAME_SPEECH))
        self.assertTrue(detector.is_speech(FRAME_SPEECH))

    def test_reset_clears_window(self):
        """reset() must empty the sample window so the next decision waits
        for a fresh 512-sample window (no cross-segment carry)."""
        detector = self._detector_with_probability(0.85)
        detector.reset()
        # Window is empty again -> first call has no decision.
        self.assertFalse(detector.is_speech(FRAME_SPEECH))

    def test_reset_clears_recurrent_state(self):
        """reset() must zero the LSTM h/c state."""
        detector = self._detector_with_probability(0.85)
        detector._h[:] = 1.0
        detector._c[:] = 1.0
        detector.reset()
        self.assertEqual(np.abs(detector._h).max(), 0.0)
        self.assertEqual(np.abs(detector._c).max(), 0.0)


class TestModelPathResolution(unittest.TestCase):
    """Environment-variable override for the model path."""

    def test_env_var_used_when_file_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            fake = f.name
        self.addCleanup(os.unlink, fake)
        with patch.dict(os.environ, {"INTERPRETE_VAD_MODEL": fake}):
            from src.vad import _resolve_model_path
            self.assertEqual(_resolve_model_path(), fake)

    def test_env_var_ignored_when_missing(self):
        with patch.dict(os.environ, {"INTERPRETE_VAD_MODEL": "/nope/model.onnx"}, clear=False):
            from src.vad import _resolve_model_path
            self.assertIsNone(_resolve_model_path())


if __name__ == "__main__":
    unittest.main()