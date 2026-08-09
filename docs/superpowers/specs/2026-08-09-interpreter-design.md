# Real-Time Interpreter & Transcription System Design

## 1. Overview
A professional-grade, air-gapped, real-time English <-> Spanish translation and transcription system for macOS (Apple Silicon). Built to act as a visual safety net for an interpreter during Customer Service calls.

## 2. Requirements & Constraints
- **Hardware**: macOS with Apple Silicon (M1).
- **Security**: 100% Offline (Air-Gapped). No external APIs. No PII leaves the machine.
- **Latency**: Total latency (Speech to Translated Text) must be under 1.5 - 2.0 seconds.
- **Audio Routing**: Must capture both physical Microphone (Interpreter) and virtual BlackHole (Client/Customer audio).
- **Hardware Acceleration**: Must leverage GPU via Metal Performance Shaders (MPS) for both ASR and NLP.

## 3. System Architecture
The system will be completely modular and asynchronous to prevent blocking the UI thread. It consists of 5 main components:

### 3.1 Audio Capture (`audio.py`)
- **Technology**: `PyAudio`, `webrtcvad` / Silero VAD.
- **Role**: Captures audio streams from both the physical microphone and BlackHole.
- **Process**: Applies Voice Activity Detection (VAD) to ignore silence, chunks the active voice segments, and pushes them to a `multiprocessing.Queue`.

### 3.2 Speech-to-Text / ASR (`transcriber.py`)
- **Technology**: `faster-whisper` (CTranslate2).
- **Role**: Converts audio chunks into text.
- **Process**: 
  - Runs in a dedicated background process.
  - Pulls audio from the queue.
  - Constrains language detection to `['en', 'es']` for ultra-fast processing.
  - Pushes the detected language and raw text to the Translator Queue.

### 3.3 Translation Engine (`translator.py`)
- **Technology**: HuggingFace `transformers`, PyTorch (`device="mps"`).
- **Models**: `Helsinki-NLP/opus-mt-en-es` and `Helsinki-NLP/opus-mt-es-en` (MarianMT).
- **Role**: Translates the transcribed text into the opposing language.
- **Process**:
  - Runs in a dedicated background process or thread.
  - Keeps both models loaded in memory persistently to avoid load times.
  - Uses the detected language from the ASR phase to route to the correct model.
  - Pushes the Final Data (Original Text + Translated Text) to the UI Queue.

### 3.4 User Interface (`ui.py`)
- **Technology**: `PyQt6`.
- **Role**: Displays the continuous transcription and translation.
- **Features**:
  - `FramelessWindowHint`: Minimalist, no title bar.
  - `WindowStaysOnTopHint`: Always visible over the CRM.
  - `TransparentForMouseEvents`: Click-through enabled.
  - Semi-transparent dark background.
  - Two vertical columns/sections: Original text and Translated text.
  - Consumes the UI Queue and updates labels dynamically via `pyqtSignal`.

### 3.5 Main Orchestrator (`main.py`)
- **Role**: Entry point of the application.
- **Process**: Initializes the queues, starts the background processes (`audio`, `transcriber`, `translator`), and launches the PyQt6 QApplication main loop.

## 4. Hardware Configuration (BlackHole)
- The user will need to install BlackHole (2ch or 16ch) via Homebrew.
- Audio MIDI Setup will be used to create a Multi-Output Device so the user can hear the client while routing the audio to BlackHole.

## 5. Deployment
- Environment management via `conda` or `venv`.
- A `requirements.txt` will be provided with exact versions compatible with Apple Silicon (MPS).
