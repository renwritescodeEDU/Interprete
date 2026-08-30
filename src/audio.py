import multiprocessing
import queue
import time
import logging
import fractions
import numpy as np
import pyaudio
import scipy.signal

from src.queueutil import put_best_effort

# Constants
CHUNK = 480  # 30ms at 16000Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_RECORDING_MINUTES = 5
# How often the audio worker re-probes for an input device while waiting.
# Keeps the system in a non-crashing "waiting" state until a microphone
# (or virtual audio device) becomes available.
DEVICE_RETRY_INTERVAL = 2.0

# Host API preference for deduplication on Windows. PortAudio exposes the same
# physical device once per host API (MME, DirectSound, WASAPI, WDM-KS). WASAPI
# is the most reliable for REAL hardware capture; MME the least. The keys are
# the PortAudio host API indices on Windows (0=MME, 1=DirectSound, 2=WASAPI,
# 3=WDM-KS).
_HOST_API_PREFERENCE = {
    2: 0,  # WASAPI (most reliable on real hardware)
    1: 1,  # DirectSound
    3: 2,  # WDM-KS
    0: 3,  # MME (least reliable on real hardware)
}
# Virtual-cable devices (VB-CABLE / Voicemeeter) are the OPPOSITE: their WASAPI
# capture endpoint often opens successfully but delivers silence (observed:
# 'CABLE Output' via WASAPI captured pure silence while the MME endpoint of the
# same device worked). For these, prefer MME/DirectSound.
_VIRTUAL_DEVICE_MARKERS = ("vb-audio", "voicemeeter", "cable")
_VIRTUAL_HOST_API_PREFERENCE = {
    0: 0,  # MME (most reliable for virtual cables)
    1: 1,  # DirectSound
    3: 2,  # WDM-KS
    2: 3,  # WASAPI (delivers silence on some virtual cables)
}

logger = logging.getLogger(__name__)

def _safe_ui_put(ui_queue, msg: dict, block: bool = False, timeout: float = 2.0):
    """Best-effort delivery of a UI event. ui_queue is optional."""
    if ui_queue is None:
        return
    put_best_effort(ui_queue, msg, block=block, timeout=timeout)


def _clean_device_name(name):
    """Repair mojibake in PortAudio device names.

    PyAudio decodes the raw name bytes with the OS preferred encoding
    (cp1252 on Spanish/European Windows) before falling back to UTF-8. Modern
    Windows devices report UTF-8 names, so UTF-8 bytes decoded as cp1252
    produce mojibake such as "MicrÃ³fono" instead of "Micrófono". Re-encoding
    the string as Latin-1 and decoding as UTF-8 restores the correct text.
    Names without mojibake pass through unchanged.
    """
    if not isinstance(name, str):
        return str(name)
    try:
        repaired = name.encode("latin-1").decode("utf-8")
        return repaired if repaired != name else name
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def _deduplicate_devices(devices):
    """Collapse host-API duplicates of the same physical device.

    PortAudio on Windows lists every device once per host API (MME,
    DirectSound, WASAPI, WDM-KS), so the same microphone appears 3-4 times
    with confusing indices. Keep a single entry per normalized name,
    preferring the most reliable host API.

    Note on name truncation: PortAudio's MME host API reports names truncated
    to 30-31 characters, while WASAPI/DirectSound report the full name.
    Without normalizing, the SAME device appears twice with slightly different
    names (observed: 'CABLE Output (VB-Audio Virtual ' MME vs 'CABLE Output
    (VB-Audio Virtual Cable)' WASAPI), and the user can pick the unreliable
    host-API duplicate. Names are matched when one is a prefix of the other
    (the truncated MME name is a prefix of the full WASAPI name).
    """
    def _rank(dev):
        name = dev["name"].lower()
        if any(marker in name for marker in _VIRTUAL_DEVICE_MARKERS):
            # VB-CABLE / Voicemeeter virtual endpoints: prefer MME/DirectSound
            # over WASAPI (WASAPI often opens but delivers silence).
            return _VIRTUAL_HOST_API_PREFERENCE.get(dev.get("host_api"), 9)
        return _HOST_API_PREFERENCE.get(dev.get("host_api"), len(_HOST_API_PREFERENCE))

    best = []
    for dev in devices:
        name = dev["name"].strip().lower()
        if not name:
            continue
        match = None
        for entry in best:
            other = entry["name"].strip().lower()
            # The truncated MME name is a prefix of the full host-API name.
            if name.startswith(other) or other.startswith(name):
                match = entry
                break
        if match is None:
            best.append(dev)
        elif _rank(dev) < _rank(match):
            # Keep the preferred host-API entry but preserve the default flag.
            dev["is_default"] = dev["is_default"] or match["is_default"]
            best[best.index(match)] = dev

    result = sorted(best, key=lambda d: d["index"])
    if len(result) != len(devices):
        logger.info(
            f"Audio device enumeration: {len(devices)} raw -> {len(result)} "
            f"unique devices (host-API duplicates collapsed)"
        )
    return result

def list_audio_devices():
    """
    Returns a list of ALL available audio devices (input, output, and both).
    Each device is a dict: {index, name, max_input_channels, max_output_channels,
                            default_sample_rate, is_default, type}
    type is one of: 'input', 'output', 'both'

    Never raises: if PortAudio cannot initialize (e.g. disabled audio service,
    VM without audio, broken drivers on Windows), returns an empty list so the
    caller can keep the app alive in a "waiting for microphone" state.
    """
    try:
        p = pyaudio.PyAudio()
    except Exception as e:
        logger.error(f"PyAudio failed to initialize: {e}")
        return []

    devices = []
    try:
        try:
            default_input_index = p.get_default_input_device_info()["index"]
        except IOError as e:
            logger.warning(f"Could not get default input device info: {e}")
            default_input_index = -1

        device_count = p.get_device_count()
        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                has_input = info["maxInputChannels"] > 0
                has_output = info["maxOutputChannels"] > 0

                if has_input and has_output:
                    dev_type = "both"
                elif has_input:
                    dev_type = "input"
                elif has_output:
                    dev_type = "output"
                else:
                    continue  # Skip devices with no channels at all

                devices.append({
                    "index": i,
                    "name": _clean_device_name(info["name"]),
                    "max_input_channels": info["maxInputChannels"],
                    "max_output_channels": info["maxOutputChannels"],
                    "default_sample_rate": info["defaultSampleRate"],
                    "is_default": (i == default_input_index),
                    "type": dev_type,
                    "host_api": info.get("hostApi"),
                })
            except Exception as e:
                logger.warning(f"Error querying device index {i}: {e}")
                continue
    finally:
        try:
            p.terminate()
        except Exception:
            pass

    return _deduplicate_devices(devices)


def _open_stream(p, device_index=None):
    """
    Opens a PyAudio input stream for the given device_index.
    Returns (stream, actual_rate, buffer_size).
    Tries the target RATE first; falls back to the device's native rate with resampling.
    Raises RuntimeError if no stream can be opened.
    """
    open_kwargs = {
        "format": FORMAT,
        "channels": CHANNELS,
        "rate": RATE,
        "input": True,
        "frames_per_buffer": CHUNK,
    }
    if device_index is not None:
        open_kwargs["input_device_index"] = device_index

    try:
        stream = p.open(**open_kwargs)
        return stream, RATE, CHUNK
    except Exception as e:
        logger.warning(f"Failed to open audio at {RATE}Hz: {e}. Attempting native rate fallback...")

    # Fallback: open at the device's native sample rate
    try:
        if device_index is not None:
            dev_info = p.get_device_info_by_index(device_index)
        else:
            dev_info = p.get_default_input_device_info()

        actual_rate = int(dev_info["defaultSampleRate"])
        buffer_size = int(CHUNK * actual_rate / RATE)

        fallback_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": actual_rate,
            "input": True,
            "frames_per_buffer": buffer_size,
        }
        if device_index is not None:
            fallback_kwargs["input_device_index"] = device_index

        stream = p.open(**fallback_kwargs)
        logger.info(f"Successfully opened audio at {actual_rate}Hz. Will resample to {RATE}Hz.")
        return stream, actual_rate, buffer_size
    except Exception as e2:
        raise RuntimeError(f"Failed to open audio stream: {e2}")


def _close_stream(stream):
    """Best-effort teardown of an audio stream. Never raises."""
    if stream is None:
        return
    try:
        stream.stop_stream()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass


def _process_audio_frames(frames_list: list, actual_rate: int) -> np.ndarray:
    """Converts raw bytes to normalized float32 array and resamples if needed."""
    audio_bytes = b"".join(frames_list)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if actual_rate != RATE:
        frac = fractions.Fraction(RATE, actual_rate)
        audio_array = scipy.signal.resample_poly(audio_array, frac.numerator, frac.denominator)

    return audio_array


def start_audio_capture(asr_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, ui_queue=None, device_index=None, final_queue: multiprocessing.Queue = None):
    """
    Captures audio strictly between START and FINISH commands,
    and pushes tuples of (audio_array, sample_rate, is_final) to asr_queue.
    Supports SET_DEVICE commands to switch input device when not recording.

    Robustness: never crashes when no microphone is available. If the input
    stream cannot be opened, the worker stays alive in a "waiting" state and
    re-probes for an input device every DEVICE_RETRY_INTERVAL seconds. Status
    transitions ("waiting" / "ready") are reported to the UI so the user sees
    a recoverable "connect a microphone" state instead of a dead pipeline.
    """
    p = None
    stream = None
    actual_rate = None
    buffer_size = None
    partial_threshold = 0
    is_recording = False
    waiting_reported = False
    last_retry = None

    frames = []
    frames_since_last_partial = 0
    partial_start_index = 0
    truncation_reported = False
    truncated_frames = 0

    recording_start_ts = None
    recording_stop_ts = None

    def _try_open_stream():
        """Attempt to (re)open the input stream; returns True on success.

        Tries the pinned device first; if it is gone (unplugged, stale index),
        falls back to the system default input device so a freshly connected
        microphone is picked up without a restart.
        """
        nonlocal p, stream, actual_rate, buffer_size, partial_threshold
        if p is None:
            try:
                p = pyaudio.PyAudio()
            except Exception as e:
                logger.warning(f"PyAudio initialization failed: {e}")
                return False

        if device_index is not None:
            try:
                stream, actual_rate, buffer_size = _open_stream(p, device_index)
                partial_threshold = int((5.0 * actual_rate) / buffer_size)
                return True
            except RuntimeError as e:
                logger.warning(f"Selected device unavailable: {e}. Trying the default input device...")
                stream = None

        try:
            stream, actual_rate, buffer_size = _open_stream(p, None)
            partial_threshold = int((5.0 * actual_rate) / buffer_size)
            return True
        except RuntimeError as e:
            logger.warning(f"Default microphone unavailable: {e}")
            stream = None
            return False

    # Initial open attempt. If it fails we enter the waiting/retry loop below.
    # Report the outcome either way so the UI's audio_ready flag tracks the
    # worker's actual state from the very first moment.
    if _try_open_stream():
        _safe_ui_put(ui_queue, {"type": "status", "process": "audio", "status": "ready"})
    else:
        waiting_reported = True
        _safe_ui_put(ui_queue, {"type": "status", "process": "audio", "status": "waiting"})

    try:
        while True:
            # Check control queue for commands
            try:
                cmd = control_queue.get_nowait()
                # Extract command name and optional timestamp from tuple format
                if isinstance(cmd, tuple) and len(cmd) == 2:
                    cmd_name, cmd_ts = cmd
                else:
                    cmd_name = cmd
                    cmd_ts = None

                if cmd_name == "QUIT":
                    break
                elif cmd_name == "START":
                    if stream is None:
                        logger.warning("START ignored: no microphone available.")
                        if not waiting_reported:
                            waiting_reported = True
                            _safe_ui_put(ui_queue, {"type": "status", "process": "audio", "status": "waiting"})
                        continue
                    is_recording = True
                    recording_start_ts = cmd_ts or time.time()
                    recording_stop_ts = None
                    frames = []
                    frames_since_last_partial = 0
                    partial_start_index = 0
                    truncation_reported = False
                    truncated_frames = 0
                    logger.info("[AUDIO] Recording started")
                    try:
                        # Flush any stale audio from hardware buffer
                        while stream.get_read_available() > 0:
                            stream.read(stream.get_read_available(), exception_on_overflow=False)
                    except Exception as e:
                        logger.warning(f"Buffer flush failed: {e}")

                elif cmd_name == "FINISH" and is_recording:
                    recording_stop_ts = cmd_ts or time.time()
                    recording_duration = recording_stop_ts - recording_start_ts if recording_start_ts else 0
                    logger.info(f"[AUDIO] Recording stopped (duration: {recording_duration:.2f}s)")

                    timing = {
                        "recording_start": recording_start_ts,
                        "recording_stop": recording_stop_ts,
                    }

                    # Clear any stale partials from the queue. multiprocessing
                    # Queue.empty() is unreliable across processes, so drain
                    # with get_nowait() until it raises.
                    while True:
                        try:
                            asr_queue.get_nowait()
                        except queue.Empty:
                            break

                    if frames:
                        audio_array = _process_audio_frames(frames, actual_rate)
                        final_msg = (audio_array, RATE, True, timing)
                        try:
                            if final_queue is not None:
                                final_queue.put(final_msg, block=True, timeout=2.0)
                            else:
                                asr_queue.put(final_msg, block=True, timeout=2.0)
                            logger.info(
                                f"[AUDIO] Final pushed to "
                                f"{'final queue' if final_queue is not None else 'ASR queue'}: "
                                f"{len(frames)} frames ({len(audio_array)} samples, {recording_duration:.2f}s)"
                            )
                        except queue.Full:
                            logger.error("queue full, dropped final audio chunk")
                    else:
                        empty_msg = (np.array([], dtype=np.float32), RATE, True, timing)
                        try:
                            if final_queue is not None:
                                final_queue.put(empty_msg, block=True, timeout=2.0)
                            else:
                                asr_queue.put(empty_msg, block=True, timeout=2.0)
                            logger.info(f"[AUDIO] Final (empty) pushed to {'final queue' if final_queue is not None else 'ASR queue'}")
                        except queue.Full:
                            logger.error("final queue full, dropped final empty chunk")

                    frames = []
                    frames_since_last_partial = 0
                    partial_start_index = 0
                    is_recording = False
                    recording_start_ts = None

                elif cmd_name == "SET_DEVICE":
                    if not is_recording:
                        new_device_index = cmd_ts
                        logger.info(f"Switching audio device to index {new_device_index}...")
                        _close_stream(stream)
                        stream = None
                        device_index = new_device_index
                        # Retry immediately on the new device (no interval wait).
                        last_retry = None
                        waiting_reported = False
                    else:
                        logger.warning("Cannot switch device while recording.")

            except queue.Empty:
                pass

            # Ensure a stream exists. If not, re-probe on a bounded interval so
            # a hot loop is avoided while the process stays responsive to
            # control commands (QUIT / SET_DEVICE).
            if stream is None:
                now = time.monotonic()
                if last_retry is None or (now - last_retry) >= DEVICE_RETRY_INTERVAL:
                    last_retry = now
                    if _try_open_stream():
                        logger.info("[AUDIO] Microphone detected; audio stream opened.")
                        waiting_reported = False
                        _safe_ui_put(ui_queue, {"type": "status", "process": "audio", "status": "ready"})
                    elif not waiting_reported:
                        waiting_reported = True
                        _safe_ui_put(ui_queue, {"type": "status", "process": "audio", "status": "waiting"})
                else:
                    time.sleep(0.05)
                continue

            # Only read and process frames if actively recording
            if is_recording:
                try:
                    frame = stream.read(buffer_size, exception_on_overflow=False)
                except Exception as e:
                    # Device removed while recording (unplugged / RDP detach).
                    logger.error(f"Audio stream read failed (device removed?): {e}")
                    _close_stream(stream)
                    stream = None
                    is_recording = False
                    waiting_reported = False
                    last_retry = None
                    continue
                frames.append(frame)
                frames_since_last_partial += 1

                max_frames = int(MAX_RECORDING_MINUTES * 60 * actual_rate / buffer_size)
                if len(frames) > max_frames:
                    dropped_frames = len(frames) - max_frames
                    frames = frames[-max_frames:]
                    partial_start_index = max(0, partial_start_index - dropped_frames)
                    truncated_frames += dropped_frames
                    if not truncation_reported:
                        truncation_reported = True
                        dropped_seconds = round(truncated_frames / actual_rate, 1)
                        logger.warning(
                            f"[AUDIO] Recording truncated at {MAX_RECORDING_MINUTES} min; "
                            f"audio beyond {MAX_RECORDING_MINUTES} min is being discarded."
                        )
                        _safe_ui_put(ui_queue, {
                            "type": "truncated",
                            "dropped_seconds": dropped_seconds,
                            "max_minutes": MAX_RECORDING_MINUTES,
                        })

                # Emit a partial transcript periodically
                if frames_since_last_partial >= partial_threshold and len(frames) > partial_start_index:
                    partial_frames = frames[partial_start_index:]
                    audio_array = _process_audio_frames(partial_frames, actual_rate)
                    try:
                        asr_queue.put((audio_array, RATE, False), block=False)
                    except queue.Full:
                        pass
                    partial_start_index = len(frames)
                    frames_since_last_partial = 0
            else:
                # Discard hardware buffer to prevent overflow while idle
                try:
                    read_avail = stream.get_read_available()
                    if read_avail >= buffer_size:
                        stream.read(read_avail, exception_on_overflow=False)
                except Exception as e:
                    logger.debug(f"Idle buffer discard failed: {e}")
                    _close_stream(stream)
                    stream = None
                    waiting_reported = False
                    last_retry = None
                    continue

    except Exception as e:
        logger.error(f"Audio capture terminated: {e}")
        _safe_ui_put(ui_queue, {"type": "error", "message": f"Audio error: {e}"}, block=True, timeout=5.0)
    finally:
        _close_stream(stream)
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
