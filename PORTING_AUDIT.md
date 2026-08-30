# Cross-Platform Porting Audit: Interprete (macOS → Windows)

**Audit date:** 2026-08-29
**Target repo:** `Interprete` (commit `b5e21bb`), cloned to `C:\Users\herre\Downloads\Interprete`
**Environment tested:** Windows 10/11, Python 3.11.9 (64-bit), git 2.48.1, Ollama CLI 0.32.15 (installed, not running)

---

## 0. Executive Summary

The application is **substantially cross-platform already**. Empirical verification on this Windows machine proved that:

- All **124/125 unit tests pass** on Windows (the 1 deselected test is a hardware-dependent integration test).
- `PyQt6`, `pyaudio`, `faster-whisper`, `ollama`, `numpy`, `scipy` all install and import cleanly on Windows.
- The full app launches: background processes spawn (audio / transcriber / translator), the PyQt6 window is created, and the UI event pipeline works (ready/error/translation events flow correctly).
- The multiprocessing design is already Windows-correct: `main()` forces `spawn` (the Windows default), and the `if __name__ == "__main__":` guard prevents child-process recursion.

The UI failure you experienced is **not a fundamental Qt-on-Windows incompatibility**. It is a combination of **environmental prerequisites** and **fragile error handling in the audio path**:

1. **No default audio input device** → the audio worker dies at startup → the UI watchdog declares a "Worker crash" and permanently disables the app. On a machine with no microphone (or a VM/RDP session), the app opens but is immediately dead — which reads as "the UI failed."
2. **`list_audio_devices()` / `pyaudio.PyAudio()` is called unguarded** during `MainWindow.__init__` (`src/ui.py:80`). On a machine where PortAudio/WASAPI fails to initialize (broken audio drivers, VM, RDP), this raises an exception **before `window.show()`**, so no window ever appears and the app exits — a literal "UI fails to display" crash.
3. **Ollama must be running** (`ollama serve`) or the translator cannot load `llama3.2:3b`; first translation then errors ("Check that Ollama is running").
4. **First run downloads models over the network** (~460 MB faster-whisper "small" + ~2 GB llama3.2:3b). The design doc claims "100% offline / air-gapped" — this is now inaccurate and is a hard blocker on air-gapped Windows machines.
5. **`WA_TranslucentBackground` + frameless windows** are a known Windows-only hazard: they can render as an invisible/black window under RDP, VMs without GPU compositing, or when desktop composition is disabled. This is the most plausible cause of "window never appears" on Windows when the code itself did not crash.

None of these are "macOS-specific dependencies" in the code — the remaining macOS artifacts are **documentation only** (`instructions_blackhole.md`, the design doc, and the Apple-only CSS font stack in `src/ui.py:98`).

---

## 1. Architecture Analysis

### 1.1 Pipeline topology

```
src/main.py  (Orchestrator — main/UI process)
│   spawns 3 daemon processes (multiprocessing, start method = "spawn")
│
├─ src/audio.py        start_audio_capture(asr_queue, control_queue, ui_queue, device_index, final_queue)
│      PyAudio capture (16 kHz mono int16), manual START/FINISH gating,
│      native-rate fallback + scipy resample, partial chunks every ~5 s,
│      final chunk on a dedicated priority queue
│
├─ src/transcriber.py  start_transcriber(asr_queue, translation_queue, ui_queue, final_queue)
│      faster-whisper "small", device="auto", compute_type="int8",
│      language detection constrained to {en, es} (confidence ≥ 0.60),
│      partial transcripts → provisional tasks; final → terminal translation task
│
├─ src/translator.py   start_translator(translation_queue, ui_queue)
│      Ollama llama3.2:3b, glossary-enhanced prompt, JSON output,
│      thread pools (3 final + 1 provisional), bilingual context history,
│      deterministic post-processing (orthography, honorifics, register)
│
└─ src/ui.py           run_ui(ui_queue, control_queue, start_cb, stop_cb, log_path, health_check)
       PyQt6 frameless, translucent, always-on-top window; 100 ms queue poller;
       3 s health watchdog; config persistence in <repo>/.config/preferences.json
```

Cross-platform verdict per module:

| Module | Cross-platform? | Notes |
|---|---|---|
| `src/main.py` | ✅ Yes | `spawn` forced; guard present; queue wiring correct on Windows |
| `src/audio.py` | ⚠️ Mostly | PortAudio init is unguarded (crash path); dies hard when no input device; no retry |
| `src/transcriber.py` | ✅ Yes | `device="auto"`, `int8` works on Windows CPU/CUDA; model auto-downloads |
| `src/translator.py` | ✅ Yes | Ollama client is HTTP-based (`127.0.0.1:11434`), OS-agnostic |
| `src/ui.py` | ⚠️ Mostly | Apple-only font stack; unguarded device refresh; translucent-window caveat on Windows |
| `src/glossary.py` | ✅ Yes | Pure stdlib; paths built with `os.path.join` from `__file__` (correct on both OSes) |
| `tests/` | ✅ Yes | 124 pass; `QT_QPA_PLATFORM=offscreen` already set for headless UI tests |
| `.github/workflows/test.yml` | ⚠️ CI gap | Runs only on `ubuntu-24.04` — Windows regressions are never caught |

### 1.2 macOS-specific artifacts found

1. **`instructions_blackhole.md`** — BlackHole is a macOS virtual audio driver installed via Homebrew (`brew install blackhole-2ch`). The **code does not hardcode BlackHole** (devices are enumerated dynamically in `src/audio.py`), so this is documentation-only. Windows needs an equivalent loopback solution (see §4).
2. **`docs/superpowers/specs/2026-08-09-interpreter-design.md`** — explicitly targets "macOS (Apple Silicon)", `device="mps"`, BlackHole, air-gapped. The implementation has already diverged (uses `device="auto"` + Ollama), so the spec is stale.
3. **`src/ui.py:98`** — `font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", ...'`. `-apple-system` and `BlinkMacSystemFont` are **CSS web font names**, not Qt font families. Qt silently ignores unknown families, so it is not fatal, but it is dead weight on Windows and should be replaced with a Qt-native approach.
4. **`.gitignore`** — contains `.DS_Store` (harmless), no Windows-specific entries needed.

### 1.3 Dependency review (`pyproject.toml`, `requirements.txt`)

| Package | Version pin | Windows status |
|---|---|---|
| `pyaudio` | `==0.2.14` | ✅ Official wheels exist for cp38–cp313 (win32 + win_amd64), PortAudio 19.7 bundled, WASAPI/MME/DirectSound/WDM-KS (no ASIO). This is no longer the classic Windows install blocker — but only because of this exact pin; do **not** bump to a version without wheels. |
| `PyQt6` | `==6.10.2` | ✅ Official Windows wheels |
| `faster-whisper` | `==1.2.1` | ✅ Pure Python + `ctranslate2` (wheels for Windows) + `av` |
| `numpy` | `>=1.24,<2.0` | ✅ |
| `ollama` | `==0.6.2` | ✅ HTTP client, OS-agnostic |
| `scipy` | `==1.13.1` | ✅ Windows wheels |

`requires-python = ">=3.10"` with **no upper bound**. Recommended runtime is 3.11/3.12; keep this documented because pyaudio 0.2.14 wheels currently cover through 3.13, and 3.14 support is not guaranteed.

---

## 2. Error Diagnosis — why the UI "failed" on Windows

Empirical run on this machine (`python -m src.main`, `QT_QPA_PLATFORM=offscreen`):

```
[INFO]  __main__: Starting background processes...
[WARNING] src.audio: Could not get default input device info: No Default Input Device Available
[ERROR]  src.audio: Failed to open audio stream: No Default Input Device Available
[ERROR]  src.ui: [UI] Pipeline error received: Audio error: Failed to open audio stream: No Default Input Device Available
[WARNING] src.translator: Failed to pre-warm Ollama: Failed to connect to Ollama ... Is Ollama running?
[ERROR]  src.ui: [UI] Worker crash detected: audio
```

Diagnostic conclusions, ranked by likelihood:

1. **Audio worker dies on missing input device (confirmed).** The machine has no default input device (only output devices). `_open_stream()` fails at both 16 kHz and native rate, raises `RuntimeError`, and `start_audio_capture` returns. Two watchdog ticks later the UI shows **"System Error — Worker crash: audio"** and disables the Start button (`src/ui.py:303-328`). The window renders, but the app is non-functional — experienced as "the UI doesn't work."

2. **Crash-before-show if PortAudio init fails (high likelihood on problem machines).** `MainWindow.__init__` → `_refresh_devices()` (`src/ui.py:228`) → `list_audio_devices()` (`src/audio.py:34`) calls `pyaudio.PyAudio()` **outside any try/except**. On macOS CoreAudio always initializes; on Windows (VMs, RDP sessions, disabled audio service, broken drivers) `PyAudio()` can raise `OSError`/`IOError`, which propagates through `_refresh_devices` → `MainWindow.__init__` → `run_ui`, so **`window.show()` is never reached** — the app crashes with a traceback and no window. This matches "UI fails to display."

3. **Invisible translucent window on RDP/VM (plausible).** `FramelessWindowHint | WindowStaysOnTopHint | WA_TranslucentBackground` (`src/ui.py:94-96`) works on real Windows desktops, but translucent frameless windows are a documented failure mode under RDP or when desktop composition/GPU acceleration is unavailable — the window exists but renders fully transparent, so "nothing shows."

4. **Ollama not running (confirmed on this machine).** `ollama.show`/`chat` fail with connection errors. The pre-warm failure is only logged; the UI shows no warning until the first translation, which then errors with "Translation failed. Check that Ollama is running."

5. **First-run model downloads.** faster-whisper pulls `small` (~460 MB) from HuggingFace; Ollama pulls `llama3.2:3b` (~2 GB). On a firewalled/air-gapped Windows box the transcriber crashes during `WhisperModel(...)`, and the watchdog then kills the UI's usefulness. The design doc's "100% Offline (Air-Gapped)" requirement is not met by the current implementation.

6. **Entry-point confusion (minor).** `python -m src.main` is correct. `python src/main.py` also works when installed with `pip install -e .` (editable install adds the repo root to `sys.path`); it breaks if run from a raw clone without the editable install. Standardize on `python -m src.main`.

7. **Python version mismatch.** If you previously used Python 3.13+ or a `python3`-style invocation (`python3 -m venv`, `source venv/bin/activate`) those fail on Windows: `python3` is not on PATH by default, and `source`/`venv/bin/activate` are POSIX-only. PowerShell also blocks `venv\Scripts\Activate.ps1` unless the execution policy allows scripts (observed on this machine).

---

## 3. Cross-Platform Verification of the Installation Steps

| Step | macOS | Windows | Verdict |
|---|---|---|---|
| `git clone ...` | `git clone` | `git clone` (same) | ✅ same |
| `python3 -m venv venv` | `python3` available | use `python` (or `py -3.11`) — `python3` usually absent | ⚠️ needs branch |
| `source venv/bin/activate` | POSIX activate | `venv\Scripts\activate` (PowerShell: `Activate.ps1` is blocked by default execution policy; workaround: `Set-ExecutionPolicy -Scope Process Bypass`, or call `venv\Scripts\python.exe` directly) | ⚠️ needs branch + policy note |
| `pip install -e ".[dev]"` | works | works (verified: PyQt6, pyaudio 0.2.14, faster-whisper, ollama all install) | ✅ same |
| `ollama pull llama3.2:3b` | `ollama serve` + pull | Ollama for Windows (GUI tray app); must be running (`ollama serve` or tray); same pull command | ✅ same (but must be running) |
| faster-whisper "small" | auto-downloads to `~/.cache/huggingface` | auto-downloads to `C:\Users\<user>\.cache\huggingface` | ✅ same behavior |
| `python -m src.main` or `interprete` | works | works (verified) | ✅ same |
| BlackHole loopback | `brew install blackhole-2ch` + Audio MIDI Setup | no BlackHole; use **Stereo Mix**, **VB-CABLE**, or **Windows "Listen to this device"** (see §5.7) | ⚠️ docs only |

**Verdict:** the pip/venv/Ollama steps are already robust on both OSes; the only real Windows installation friction is the PowerShell execution policy and the `python3`/`source` syntax.

---

## 4. Action Plan (step-by-step implementation)

### Phase A — Code hardening (crash + robustness fixes)

**A1. Guard the PyAudio initialization in `src/audio.py:list_audio_devices()`**
Wrap `p = pyaudio.PyAudio()` (line 34) in `try/except` and return `[]` on failure instead of raising. This removes the "crash before window shows" path.

```python
def list_audio_devices():
    try:
        p = pyaudio.PyAudio()
    except Exception as e:
        logger.error(f"PyAudio failed to initialize: {e}")
        return []
    ...
```

**A2. Don't let a missing input device kill the app — `src/ui.py`**
In `_refresh_devices()`:
- If `list_audio_devices()` returns `[]` (or returns no input-capable device), show a clear status line instead of silently selecting an output device:
  `"No microphone found — connect one and press Refresh"`.
- Currently, with no default input device, `selected_combo_index` stays `0`, which is an **output-only** device → clicking Start fails. Filter the combo to input-capable devices (`type in {"input", "both"}`) and warn when the list is empty.

**A3. Make the audio worker resilient instead of dying (`src/audio.py:start_audio_capture`)**
On Windows, audio devices can appear/disappear (plug/unplug, RDP attach). Replace the immediate `return` after a stream-open failure with a bounded retry loop (e.g., retry every 3 s for up to ~60 s), and emit one clear `{"type": "error", "message": "No input device found..."}` **only after** retries are exhausted. This prevents the watchdog from showing "Worker crash" for a recoverable condition and gives the user actionable feedback.

**A4. Soften the watchdog (`src/ui.py:check_health`)**
The 2-miss crash declaration is correct for transcriber/translator, but an audio worker that failed cleanly (stream open error) should not flip the whole UI into a permanent dead state. Track *why* a worker stopped: have the audio process send a terminal `{"type": "audio_unavailable"}` event on clean failure; the watchdog skips the crash banner for that specific case and shows the "no microphone" hint instead. Optionally allow the user to re-select a device and restart the audio worker without restarting the app.

**A5. Surface Ollama status in the UI (`src/translator.py`)**
The pre-warm failure is currently logged only. Send a `{"type": "status", "process": "translator", "status": "ollama_offline"}` event when `ollama.show`/`chat` raises a connection error, and have the UI show `"Ollama not running — start it and pull llama3.2:3b"` in the status label while keeping the app usable.

**A6. Fix the font stack (`src/ui.py:98`)**
Remove the Apple-only CSS names. Use a Qt-native approach:

```python
from PyQt6.QtGui import QFont
...
font_family = QFont().defaultFamily()          # resolves "Segoe UI" on Windows, system font on macOS
```
and reference `font_family` in the stylesheet as before. This guarantees a valid family on both OSes.

**A7. Guard `QGuiApplication.primaryScreen()` (`src/ui.py:287-293`)**
Already null-checked, but if `screen` is `None` the window is never centered; also add a fallback `self.move(0, 0)`. Low priority, one line.

**A8. Resolve the air-gap / first-run download problem**
Document (and optionally implement) explicit model pre-fetch:
- `src/transcriber.py`: keep auto-download, but log the model path so users can pre-download it; optionally add a `INTERPRETE_WHISPER_MODEL` env override for offline use.
- Add a `setup_offline.md` (or extend the README) describing how to pre-pull models on an internet-connected machine and copy the caches (HuggingFace cache + `ollama pull` cache) to the offline Windows box.

### Phase B — CI and documentation

**B1. Add Windows to CI (`.github/workflows/test.yml`)**
Add `windows-latest` to the `runs-on` matrix (or a second job). PyAudio wheels make this possible; `QT_QPA_PLATFORM=offscreen` is already used in tests so no display is needed. This catches future Windows regressions automatically.

**B2. Rewrite `instructions_blackhole.md` → split into `docs/audio_routing.md`**
Keep macOS/BlackHole steps, add a Windows section:

| Windows loopback option | Setup |
|---|---|
| **Stereo Mix** (Realtek) | Sound → Recording → enable "Stereo Mix" (right-click → Show Disabled Devices) → set as default input |
| **VB-CABLE** (free virtual audio cable) | Install → select "CABLE Input" as the app/CRM output, "CABLE Output" as the Interprete input |
| **Windows "Listen to this device"** | Record tab → device → Properties → Listen → "Listen to this device" → playback through speakers; capture the mic in Interprete |

**B3. Write `README.md` / `INSTALL.md` with OS-specific commands**
- macOS: `python3 -m venv venv && source venv/bin/activate && pip install -e ".[dev]"`
- Windows: `python -m venv venv` then either `Set-ExecutionPolicy -Scope Process Bypass; .\venv\Scripts\Activate.ps1` or `.\venv\Scripts\python.exe -m pip install -e ".[dev]"`, then `python -m src.main`
- Both: `ollama serve` (or tray) **before** launching; `ollama pull llama3.2:3b`; first run downloads faster-whisper "small".
- Prerequisites checklist: working microphone (verify in Windows Sound settings), Ollama running, network for first-run model downloads.

**B4. Refresh the stale design spec**
Update `docs/superpowers/specs/2026-08-09-interpreter-design.md` to drop the macOS-only claims (MPS, air-gap as a hard requirement, BlackHole-only routing) or explicitly mark them "macOS notes."

### Phase C — Optional enhancements

- **C1. Microphone permission check on Windows 11** — Windows may block mic access for the terminal process; add a friendly hint if the first stream-open fails with an access error (`[Errno -9988] Invalid device` or WASAPI permission errors).
- **C2. Dependency floor guard** — pin `pyaudio==0.2.14` comment explaining why (Windows wheels cp38–cp313) so a future bump doesn't silently break Windows installs.
- **C3. Auto-restart of the audio worker** when the user switches to a newly-available device (ties into A3/A4).

---

## 5. Specific Windows Setup Instructions (after Phase A/B changes, or immediately)

1. **Install Python 3.11 or 3.12 (64-bit)** from python.org — check "Add python.exe to PATH".
2. **Install Ollama for Windows** (https://ollama.com/download) and start it (tray app). Verify: `ollama list` (empty is fine) then `ollama pull llama3.2:3b` (first run, ~2 GB).
3. **Install Git for Windows** (git-scm.com) if not present.
4. Clone and set up:
   ```powershell
   cd C:\Users\herre\Downloads
   git clone https://github.com/renwritescodeEDU/Interprete.git
   cd Interprete
   python -m venv venv
   Set-ExecutionPolicy -Scope Process Bypass   # allows venv activation scripts
   .\venv\Scripts\Activate.ps1                  # or use .\venv\Scripts\python.exe directly
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```
5. **Confirm a microphone exists**: Windows Settings → System → Sound → Input. If none, the app will show "No microphone found" (after Phase A2/A3) instead of dying.
6. **Verify audio loopback** if you need to capture the client's voice (not just your mic): see §B2 table (Stereo Mix / VB-CABLE / Listen-to-device).
7. **First run**: `python -m src.main` — faster-whisper downloads "small" (~460 MB) into `C:\Users\<user>\.cache\huggingface`; wait for "System Ready", then Start Recording.
8. **Run the test suite**: `python -m pytest -v -m "not slow"` (expect 124 passed).

---

## 6. Suggested commit sequence (matches repo's conventional-commit style)

1. `fix(audio): guard PyAudio init and retry stream open on missing input device`  — A1, A3
2. `fix(ui): filter input devices, warn on no microphone, soften watchdog`  — A2, A4
3. `fix(translator): surface Ollama offline status to the UI`  — A5
4. `fix(ui): use Qt-native font family`  — A6, A7
5. `ci: run tests on windows-latest`  — B1
6. `docs: add Windows install and audio-routing instructions`  — B2, B3, B4
7. `fix(transcriber): allow offline model path override`  — A8 (optional)

---

## 7. Implementation status (applied 2026-08-29)

The items below have been implemented and verified on Windows (Python 3.11.9,
no microphone present — the "waiting" path was exercised live):

| Item | Status | Where |
|---|---|---|
| A1 — guard `PyAudio()` init in `list_audio_devices()` | ✅ Done | `src/audio.py` |
| A2 — filter combo to input-capable devices; warn on no mic | ✅ Done | `src/ui.py:_refresh_devices` |
| A3 — audio worker waits & re-probes instead of dying | ✅ Done | `src/audio.py:start_audio_capture` (2 s probe interval; recovers on plug-in; stream-loss handled mid-recording) |
| A4 — watchdog skips crash banner for waiting audio | ✅ Done | `src/ui.py:check_health` |
| A5 — Ollama lifecycle: auto-start `ollama serve`, bounded wait, self-healing watcher, UI statuses | ✅ Done | `src/translator.py` (`_ensure_ollama_running`, `_warmup_ollama`, watcher thread); `src/ui.py` (ollama_waiting / ollama_offline / model_download) |
| A6 — Qt-native font family | ✅ Done | `src/ui.py:_setup_ui` |
| A7 — primaryScreen fallback | ✅ Done | `src/ui.py:_center_on_screen` |
| A8 — offline model pre-fetch doc | ⏳ Not applied | see §4 A8 |
| B1 — Windows CI matrix | ✅ Done | `.github/workflows/test.yml` (ubuntu-24.04 + windows-latest × 3.10/3.11) |
| B2 — Windows audio routing doc | ✅ Done | `docs/audio_routing.md` (VB-CABLE, Stereo Mix, Listen-to-this-device, VAC) |
| B3 — cross-platform README | ✅ Done | `README.md` |
| B4 — spec refresh | ⏳ Not applied | optional |

**New behavior added beyond the original plan:**
- UI auto-polls for devices every 3 s while no microphone is present
  (`DEVICE_POLL_INTERVAL_MS`, `_poll_devices`) and immediately points the audio
  worker at a newly detected device.
- Audio worker reports `waiting`/`ready` status transitions to the UI; the UI
  shows a recoverable "Waiting for microphone… / Connecting to microphone…"
  state instead of "System Error".
- `ollama serve` is spawned at most once per process lifetime
  (`_ollama_start_attempted` guard), with a self-healing background thread that
  flips the translator to `ready` when the server comes up later.

**Verification:**
- `pytest -m "not slow"` → 125 passed (124 pre-existing + 1 new
  `test_audio_capture_detects_device_later`; the old
  `test_audio_capture_reports_error_to_ui_queue` was replaced by
  `test_audio_capture_waits_when_no_device`).
- `pyflakes src/ tests/` → clean.
- Live run on a mic-less Windows machine: workers spawn, audio worker idles in
  waiting state (no crash banner), Ollama auto-starts and connects in ~4.5 s,
  model downloads begin. `pytest` translator tests now stub `_ollama_ready` in
  `setUp` so they never spawn a real `ollama serve`.

**Remaining gaps for full feature parity:** A8 (documented model pre-fetch for
air-gapped setups) and B4 (spec refresh) were not applied — they are
documentation-only and non-blocking.

### 7.1 Post-review fixes (applied 2026-08-29)

A code review of the implementation surfaced the following issues, all fixed:

1. **CRITICAL — missing `ready` status on initial stream open** (`src/audio.py`):
   the worker only reported `ready` from the retry path, so on a normal startup
   with a working mic the UI's `audio_ready` flag never set and the Start button
   stayed disabled. The initial-success path now emits the same `ready` event.
2. **Mid-recording device loss** (`src/audio.py`, `src/ui.py`): the worker now
   falls back to the default input device when the pinned device index is stale,
   and the UI resets its recording state when the worker reports `waiting`
   mid-recording (no more stuck "Stop Recording"/"Translating…" state).
3. **Stale device list blocking auto-recovery** (`src/ui.py`): the audio
   `waiting` status now invalidates the cached device list so the poll timer
   re-enumerates and finds newly connected microphones.
4. **Crash masking** (`src/ui.py`): removed the `_audio_waiting` guard in the
   watchdog — a genuinely dead worker now always surfaces a crash banner.
5. **Stale waiting banner** (`src/ui.py`): the "Waiting for microphone…" banner
   is cleared when the worker confirms a stream (`ready`).
6. **Dead `stop_event`** (`src/translator.py`): the Ollama watcher's stop event
   is now signaled on translator shutdown instead of being an unconditional
   daemon loop.
7. **Model-name drift** (`src/translator.py`, `src/ui.py`): the download status
   event now carries `model: LLM_MODEL` and the UI renders it dynamically.

The UI-side device polling timer was **retained** deliberately: it implements the
explicit requirement of periodically refreshing the device list while no
microphone is present; its cost (PyAudio enumeration on the main thread) is
confined to the no-input-device state it exists to recover from.
