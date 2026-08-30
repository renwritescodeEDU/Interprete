# Interprete — Simultaneous Interpreter

Real-time, offline English ⇄ Spanish interpretation for professional calls. Captures
audio, transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
and translates it locally with [Ollama](https://ollama.com) (`llama3.2:3b`). No cloud APIs
are used for speech or translation.

Runs on **macOS** and **Windows** (Python 3.10+; 3.11/3.12 recommended).

## Prerequisites

- Python 3.11 or 3.12 (64-bit)
- [Ollama](https://ollama.com/download) installed
- A working microphone (and optionally a virtual audio device — see
  [`docs/audio_routing.md`](docs/audio_routing.md))

## Installation

### macOS / Linux

```bash
git clone https://github.com/renwritescodeEDU/Interprete.git
cd Interprete
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows (PowerShell)

```powershell
git clone https://github.com/renwritescodeEDU/Interprete.git
cd Interprete
python -m venv venv
Set-ExecutionPolicy -Scope Process Bypass      # allows venv activation scripts
.\venv\Scripts\Activate.ps1                     # or use .\venv\Scripts\python.exe directly
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

> If activating the venv is not desired, prefix every command with `.\venv\Scripts\python.exe`.

## Running

```bash
python -m src.main
# or, once installed:  interprete
```

On first run the app downloads two models (needs internet):

- faster-whisper `small` (~460 MB) from HuggingFace into `~/.cache/huggingface`
- `llama3.2:3b` (~2 GB) via `ollama pull llama3.2:3b`

**Ollama is managed automatically:** if the server is not running when Interprete starts,
the app launches `ollama serve` and waits for it to become ready, retrying in the
background until it does. The UI shows "Starting Ollama…" / "Downloading llama3.2:3b…"
during this phase.

**Microphone handling:** if no input device is present at launch, the app does not crash —
it shows "Waiting for microphone…" and re-probes every few seconds. Plug in a microphone
(or a virtual audio device) and it is picked up automatically.

## Capturing call audio (loopback)

To interpret the client's side of the call you must route the call audio (CRM, browser,
phone app) into the system as an input device:

- **macOS:** install BlackHole (see `instructions_blackhole.md`)
- **Windows:** use VB-CABLE, Stereo Mix, or Windows "Listen to this device"
  (see [`docs/audio_routing.md`](docs/audio_routing.md))

## Testing

```bash
python -m pytest -v -m "not slow"     # unit tests (no hardware required)
python -m pytest -m slow              # integration/latency tests (requires mic + Ollama)
```
