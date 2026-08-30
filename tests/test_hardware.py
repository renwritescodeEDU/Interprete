"""Tests for the hardware-aware model manager (Phase 7).

Certifies that hardware detection NEVER raises (always degrades to CPU) and
that the model selection tiers + user overrides behave exactly as specified.
"""

import unittest
from unittest.mock import MagicMock, patch

from src import hardware
from src.hardware import (
    DEFAULT_LLM_MODEL,
    HEAVY_LLM_MODEL,
    HardwareProfile,
    get_hardware_profile,
    select_llm_model,
    select_whisper_config,
)


class _HardwareBase(unittest.TestCase):
    """Cache-aware base: the profile is computed once per process."""

    def setUp(self):
        get_hardware_profile.cache_clear()

    def tearDown(self):
        get_hardware_profile.cache_clear()


class TestHardwareDetection(_HardwareBase):
    @patch("src.hardware._cuda_runtime_loadable", return_value=False)
    def test_profile_cpu_when_runtime_missing(self, mock_runtime):
        """No loadable CUDA runtime -> CPU, never an exception."""
        profile = get_hardware_profile()
        self.assertEqual(profile.device, "cpu")
        self.assertFalse(profile.cuda_available)

    @patch("src.hardware._cuda_runtime_loadable", return_value=True)
    @patch("src.hardware._cuda_device_count", return_value=0)
    def test_profile_cpu_when_no_cuda_devices(self, mock_count, mock_runtime):
        profile = get_hardware_profile()
        self.assertEqual(profile.device, "cpu")
        self.assertEqual(profile.vram_gb, 0.0)

    @patch("src.hardware._cuda_runtime_loadable", return_value=True)
    @patch("src.hardware._cuda_device_count", return_value=1)
    @patch("src.hardware._query_vram_gb", return_value=12.0)
    def test_profile_cuda_with_vram(self, mock_vram, mock_count, mock_runtime):
        profile = get_hardware_profile()
        self.assertEqual(profile.device, "cuda")
        self.assertTrue(profile.cuda_available)
        self.assertEqual(profile.vram_gb, 12.0)

    @patch("src.hardware._cuda_runtime_loadable", return_value=True)
    @patch("src.hardware._cuda_device_count", return_value=1)
    @patch("src.hardware._query_vram_gb", return_value=0.0)
    def test_profile_cuda_unknown_vram(self, mock_vram, mock_count, mock_runtime):
        """CUDA without measurable VRAM still reports cuda (unknown VRAM)."""
        profile = get_hardware_profile()
        self.assertEqual(profile.device, "cuda")
        self.assertEqual(profile.vram_gb, 0.0)

    @patch("src.hardware._cuda_runtime_loadable", return_value=True)
    @patch("src.hardware._cuda_device_count", return_value=1)
    @patch("src.hardware._query_vram_gb", return_value=12.0)
    def test_profile_cached(self, mock_vram, mock_count, mock_runtime):
        """The profile is computed once per process."""
        get_hardware_profile()
        get_hardware_profile()
        mock_runtime.assert_called_once()
        mock_count.assert_called_once()
        mock_vram.assert_called_once()


class TestVramQuery(_HardwareBase):
    @patch("src.hardware.subprocess.run")
    def test_vram_parsed_from_nvidia_smi(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="12288\n")
        self.assertEqual(hardware._query_vram_gb(), 12.0)

    @patch("src.hardware.subprocess.run", side_effect=FileNotFoundError)
    def test_vram_unknown_when_nvidia_smi_missing(self, mock_run):
        self.assertEqual(hardware._query_vram_gb(), 0.0)

    @patch("src.hardware.subprocess.run")
    def test_vram_zero_on_nonzero_returncode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        self.assertEqual(hardware._query_vram_gb(), 0.0)


class TestWhisperSelection(_HardwareBase):
    @patch("src.hardware.get_hardware_profile", return_value=HardwareProfile("cpu", 0.0, False))
    def test_cpu_uses_small_int8(self, mock_profile):
        cfg = select_whisper_config(prefs={})
        self.assertEqual(cfg, {"model": "small", "compute_type": "int8", "device": "cpu"})

    @patch("src.hardware.get_hardware_profile", return_value=HardwareProfile("cuda", 12.0, True))
    def test_cuda_high_vram_medium_float16(self, mock_profile):
        """GPU >= 8 GB -> medium + float16 (tier 1)."""
        cfg = select_whisper_config(prefs={})
        self.assertEqual(cfg, {"model": "medium", "compute_type": "float16", "device": "cuda"})

    @patch("src.hardware.get_hardware_profile", return_value=HardwareProfile("cuda", 6.0, True))
    def test_cuda_mid_vram_small_float16(self, mock_profile):
        """GPU 4-8 GB -> small + float16 (tier 2)."""
        cfg = select_whisper_config(prefs={})
        self.assertEqual(cfg, {"model": "small", "compute_type": "float16", "device": "cuda"})

    @patch("src.hardware.get_hardware_profile", return_value=HardwareProfile("cuda", 3.0, True))
    def test_cuda_low_vram_small_int8(self, mock_profile):
        """GPU < 4 GB -> small + int8 (tier 3)."""
        cfg = select_whisper_config(prefs={})
        self.assertEqual(cfg, {"model": "small", "compute_type": "int8", "device": "cuda"})

    def test_user_override_whisper_wins(self):
        """Explicit user models must beat auto-detection."""
        prefs = {"whisper_model": "base", "whisper_compute_type": "int8"}
        with patch("src.hardware.get_hardware_profile", return_value=HardwareProfile("cuda", 12.0, True)):
            cfg = select_whisper_config(prefs=prefs)
        self.assertEqual(cfg["model"], "base")
        self.assertEqual(cfg["compute_type"], "int8")

    def test_user_override_whisper_default_compute(self):
        """whisper_model without whisper_compute_type keeps the int8 default."""
        prefs = {"whisper_model": "large-v3"}
        cfg = select_whisper_config(prefs=prefs)
        self.assertEqual(cfg["model"], "large-v3")
        self.assertEqual(cfg["compute_type"], "int8")


class TestLLMSelection(_HardwareBase):
    @patch("src.hardware.get_hardware_profile", return_value=HardwareProfile("cpu", 0.0, False))
    def test_cpu_uses_default_llm(self, mock_profile):
        self.assertEqual(select_llm_model(prefs={}), DEFAULT_LLM_MODEL)

    @patch("src.hardware.get_hardware_profile", return_value=HardwareProfile("cuda", 16.0, True))
    def test_cuda_high_vram_heavy_llm(self, mock_profile):
        """GPU >= 8 GB -> the heavier, idiom-capable model."""
        self.assertEqual(select_llm_model(prefs={}), HEAVY_LLM_MODEL)

    @patch("src.hardware.get_hardware_profile", return_value=HardwareProfile("cuda", 6.0, True))
    def test_cuda_low_vram_default_llm(self, mock_profile):
        """GPU < 8 GB -> keep the 3B model."""
        self.assertEqual(select_llm_model(prefs={}), DEFAULT_LLM_MODEL)

    def test_user_override_llm_wins(self):
        self.assertEqual(select_llm_model(prefs={"llm_model": "qwen2.5:7b"}), "qwen2.5:7b")


if __name__ == "__main__":
    unittest.main()