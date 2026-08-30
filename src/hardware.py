"""Hardware detection and dynamic model selection (Phase 7).

Central module that answers two questions:

1. **What hardware is available?** — is the CUDA 12 runtime loadable, and how
   much VRAM (GB) does the primary GPU report?
2. **Which models should we run?** — the faster-whisper configuration and the
   Ollama LLM, scaled to the detected hardware.

Everything here is deliberately safe: any detection failure degrades to the
CPU fallback and is logged clearly, so the pipeline ALWAYS boots. User
overrides in ``.config/preferences.json`` take precedence over auto-scaling.

Tiers (as specified):
* Whisper: GPU >= 8 GB -> ``medium`` + ``float16``; GPU 4-8 GB -> ``small`` +
  ``float16``; CPU or GPU < 4 GB -> ``small`` + ``int8``.
* Ollama: GPU >= 8 GB VRAM -> ``llama3.1:8b``; otherwise ``llama3.2:3b``.
"""

import ctypes
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache

from src.config import load_preferences

logger = logging.getLogger(__name__)

# Whisper VRAM tiers (GB)
VRAM_TIER_HEAVY = 8.0   # GPU >= 8 GB -> medium + float16
VRAM_TIER_MID = 4.0     # GPU >= 4 GB -> small + float16
# Ollama heavy-model threshold (GB of VRAM)
LLM_HEAVY_VRAM_GB = 8.0

# Absolute fallbacks
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_WHISPER_COMPUTE = "int8"
DEFAULT_LLM_MODEL = "llama3.2:3b"
HEAVY_LLM_MODEL = "llama3.1:8b"


def _dlls_loadable(libs):
    """True if every DLL in `libs` can be loaded from the current search path."""
    for lib in libs:
        try:
            ctypes.windll.LoadLibrary(lib)
        except Exception:
            return False
    return True


def _locate_cuda_bin():
    """Locate the CUDA 12 Toolkit bin directory (where cublas64_12.dll lives).

    Checks the CUDA_PATH* environment variables set by the NVIDIA installer
    first, then scans the default install root for the newest v12.x. Returns
    the bin path as a string, or None if not found.
    """
    import glob

    candidates = []
    for var in (
        "CUDA_PATH", "CUDA_PATH_V12_8", "CUDA_PATH_V12_7", "CUDA_PATH_V12_6",
        "CUDA_PATH_V12_5", "CUDA_PATH_V12_4", "CUDA_PATH_V12_3", "CUDA_PATH_V12_2",
        "CUDA_PATH_V12_1", "CUDA_PATH_V12_0",
    ):
        val = os.environ.get(var)
        if val:
            candidates.append(os.path.join(val, "bin"))

    if sys.platform == "win32":
        root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        for d in sorted(glob.glob(os.path.join(root, "v12.*")), reverse=True):
            candidates.append(os.path.join(d, "bin"))

    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "cublas64_12.dll")):
            return cand
    return None


def _cuda_runtime_loadable() -> bool:
    """True when the CUDA 12 runtime DLLs are loadable (Windows) or the
    platform has no CUDA runtime requirement (Linux/other)."""
    if sys.platform == "darwin":
        return False
    if sys.platform == "win32":
        required = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")
        if not _dlls_loadable(required):
            cuda_bin = _locate_cuda_bin()
            if cuda_bin and hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(cuda_bin)
                    logger.info(
                        f"[HARDWARE] Added CUDA Toolkit bin to the DLL search path: {cuda_bin}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[HARDWARE] Could not add {cuda_bin} to the DLL search path: {e}"
                    )
            if not _dlls_loadable(required):
                logger.warning(
                    "[HARDWARE] CUDA 12 runtime not loadable — falling back to CPU."
                )
                return False
    return True


def _cuda_device_count() -> int:
    """Number of CUDA devices visible to CTranslate2 (0 on any failure)."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count()
    except Exception as e:
        logger.warning(f"[HARDWARE] CUDA probe failed ({e}) — treating as CPU.")
        return 0


def _query_vram_gb() -> float:
    """Total VRAM in GB via nvidia-smi. Returns 0.0 when unavailable."""
    if sys.platform == "darwin":
        return 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0,
        )
        if out.returncode != 0:
            return 0.0
        match = re.search(r"(\d+)", out.stdout)
        if not match:
            return 0.0
        return float(match.group(1)) / 1024.0
    except Exception as e:
        logger.warning(f"[HARDWARE] nvidia-smi VRAM query failed ({e}) — VRAM unknown.")
        return 0.0


@dataclass(frozen=True)
class HardwareProfile:
    device: str       # "cuda" | "cpu"
    vram_gb: float    # total VRAM in GB, 0.0 if unknown
    cuda_available: bool


@lru_cache(maxsize=1)
def get_hardware_profile() -> HardwareProfile:
    """Detect the hardware once per process. NEVER raises — always returns a profile."""
    if sys.platform == "darwin":
        logger.info("[HARDWARE] macOS detected — CPU only (no CUDA backend).")
        return HardwareProfile(device="cpu", vram_gb=0.0, cuda_available=False)

    if _cuda_runtime_loadable() and _cuda_device_count() > 0:
        vram = _query_vram_gb()
        if vram > 0:
            logger.info(f"[HARDWARE] CUDA detected with {vram:.1f} GB VRAM — using GPU.")
        else:
            logger.info("[HARDWARE] CUDA detected (VRAM unknown — assuming capable).")
        return HardwareProfile(device="cuda", vram_gb=vram, cuda_available=True)

    logger.info("[HARDWARE] No usable CUDA device — running on CPU.")
    return HardwareProfile(device="cpu", vram_gb=0.0, cuda_available=False)


def select_whisper_config(prefs: dict = None) -> dict:
    """Choose the faster-whisper model/compute_type/device for this machine.

    Precedence: user override (``whisper_model`` / ``whisper_compute_type``)
    > hardware auto-scaling. Always returns a usable config.
    """
    if prefs is None:
        prefs = load_preferences()

    if prefs.get("whisper_model"):
        model = str(prefs["whisper_model"])
        compute = str(prefs.get("whisper_compute_type") or DEFAULT_WHISPER_COMPUTE)
        device = "cuda" if get_hardware_profile().device == "cuda" else "cpu"
        logger.info(
            f"[WHISPER] User override: model={model}, compute_type={compute}, device={device}"
        )
        return {"model": model, "compute_type": compute, "device": device}

    profile = get_hardware_profile()
    if profile.device == "cuda" and profile.vram_gb >= VRAM_TIER_HEAVY:
        cfg = {"model": "medium", "compute_type": "float16", "device": "cuda"}
        tier = f"tier1 (GPU {profile.vram_gb:.1f} GB VRAM)"
    elif profile.device == "cuda" and profile.vram_gb >= VRAM_TIER_MID:
        cfg = {"model": "small", "compute_type": "float16", "device": "cuda"}
        tier = f"tier2 (GPU {profile.vram_gb:.1f} GB VRAM)"
    else:
        device = "cuda" if profile.cuda_available else "cpu"
        cfg = {"model": DEFAULT_WHISPER_MODEL, "compute_type": DEFAULT_WHISPER_COMPUTE, "device": device}
        tier = "tier3 (CPU or low-VRAM GPU)"
    logger.info(
        f"[WHISPER] Auto-selected: model={cfg['model']}, "
        f"compute_type={cfg['compute_type']}, device={cfg['device']} ({tier})"
    )
    return cfg


def select_llm_model(prefs: dict = None) -> str:
    """Choose the Ollama LLM for this machine.

    Precedence: user override (``llm_model``) > heavy model on GPU >= 8 GB
    VRAM > 3B default.
    """
    if prefs is None:
        prefs = load_preferences()

    if prefs.get("llm_model"):
        model = str(prefs["llm_model"])
        logger.info(f"[OLLAMA] User override: llm_model={model}")
        return model

    profile = get_hardware_profile()
    if profile.device == "cuda" and profile.vram_gb >= LLM_HEAVY_VRAM_GB:
        logger.info(
            f"[OLLAMA] Auto-selected heavy model {HEAVY_LLM_MODEL} "
            f"(GPU {profile.vram_gb:.1f} GB VRAM)."
        )
        return HEAVY_LLM_MODEL

    logger.info(f"[OLLAMA] Auto-selected model {DEFAULT_LLM_MODEL} (CPU or modest GPU).")
    return DEFAULT_LLM_MODEL
