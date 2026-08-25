import multiprocessing
import queue
import logging
import numpy as np
import pyaudio
import scipy.signal

# Constants
CHUNK = 480  # 30ms at 16000Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_RECORDING_MINUTES = 5

logger = logging.getLogger(__name__)

def list_audio_devices():
    """
    Returns a list of ALL available audio devices (input, output, and both).
    Each device is a dict: {index, name, max_input_channels, max_output_channels,
                            default_sample_rate, is_default, type}
    type is one of: 'input', 'output', 'both'
    """
    p = pyaudio.PyAudio()
    devices = []
    
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
                "name": info["name"],
                "max_input_channels": info["maxInputChannels"],
                "max_output_channels": info["maxOutputChannels"],
                "default_sample_rate": info["defaultSampleRate"],
                "is_default": (i == default_input_index),
                "type": dev_type,
            })
        except Exception as e:
            logger.warning(f"Error querying device index {i}: {e}")
            continue

    p.terminate()
    return devices


def _open_stream(p, device_index=None):
    """
    Opens a PyAudio input stream for the given device_index.
    Returns (stream, actual_rate, buffer_size).
    Tries the target RATE first; falls back to the device's native rate with resampling.
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


def _process_audio_frames(frames_list: list, actual_rate: int) -> np.ndarray:
    """Converts raw bytes to normalized float32 array and resamples if needed."""
    audio_bytes = b"".join(frames_list)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    if actual_rate != RATE:
        import fractions
        frac = fractions.Fraction(RATE, actual_rate)
        audio_array = scipy.signal.resample_poly(audio_array, frac.numerator, frac.denominator)
        
    return audio_array


def start_audio_capture(asr_queue: multiprocessing.Queue, control_queue: multiprocessing.Queue, ui_queue=None, device_index=None):
    """
    Captures audio strictly between START and FINISH commands,
    and pushes tuples of (audio_array, sample_rate, is_final) to asr_queue.
    Supports SET_DEVICE commands to switch input device when not recording.
    """
    p = pyaudio.PyAudio()

    try:
        stream, actual_rate, buffer_size = _open_stream(p, device_index)
    except RuntimeError as e:
        logger.error(str(e))
        p.terminate()
        return

    frames = []
    partial_threshold = int((1.0 * actual_rate) / buffer_size)  
    frames_since_last_partial = 0
    partial_start_index = 0
    is_recording = False

    try:
        while True:
            # Check control queue for commands
            try:
                cmd = control_queue.get_nowait()
                if cmd == "QUIT":
                    break
                elif cmd == "START":
                    is_recording = True
                    frames = []
                    frames_since_last_partial = 0
                    partial_start_index = 0
                    try:
                        # Flush any stale audio from hardware buffer
                        while stream.get_read_available() > 0:
                            stream.read(stream.get_read_available(), exception_on_overflow=False)
                    except Exception as e:
                        logger.warning(f"Buffer flush failed: {e}")
                
                elif cmd == "FINISH" and is_recording:
                    # Clear any stale partials from the queue
                    while not asr_queue.empty():
                        try:
                            asr_queue.get_nowait()
                        except queue.Empty:
                            break

                    if frames:
                        audio_array = _process_audio_frames(frames, actual_rate)
                        try:
                            asr_queue.put((audio_array, RATE, True), block=True, timeout=2.0)
                        except queue.Full:
                            logger.error("asr_queue full, dropped final audio chunk")
                    else:
                        try:
                            asr_queue.put((np.array([], dtype=np.float32), RATE, True), block=True, timeout=2.0)
                        except queue.Full:
                            logger.error("asr_queue full, dropped final empty chunk")
                            
                    frames = []
                    frames_since_last_partial = 0
                    partial_start_index = 0
                    is_recording = False

                elif isinstance(cmd, tuple) and len(cmd) == 2 and cmd[0] == "SET_DEVICE":
                    if not is_recording:
                        new_device_index = cmd[1]
                        logger.info(f"Switching audio device to index {new_device_index}...")
                        try:
                            stream.stop_stream()
                            stream.close()
                        except Exception as e:
                            logger.warning(f"Stream close warning: {e}")
                            
                        try:
                            stream, actual_rate, buffer_size = _open_stream(p, new_device_index)
                            partial_threshold = int((1.0 * actual_rate) / buffer_size)
                            device_index = new_device_index
                        except RuntimeError as e:
                            logger.error(f"Failed to switch device: {e}. Reopening previous...")
                            try:
                                stream, actual_rate, buffer_size = _open_stream(p, device_index)
                                partial_threshold = int((1.0 * actual_rate) / buffer_size)
                            except RuntimeError:
                                logger.critical("FATAL: Cannot reopen any audio device.")
                                break
                    else:
                        logger.warning("Cannot switch device while recording.")

            except queue.Empty:
                pass

            # Only read and process frames if actively recording
            if is_recording:
                frame = stream.read(buffer_size, exception_on_overflow=False)
                frames.append(frame)
                frames_since_last_partial += 1
                
                max_frames = int(MAX_RECORDING_MINUTES * 60 * actual_rate / buffer_size)
                if len(frames) > max_frames:
                    dropped_frames = len(frames) - max_frames
                    frames = frames[-max_frames:]
                    partial_start_index = max(0, partial_start_index - dropped_frames)
                
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

    except Exception as e:
        logger.error(f"Audio capture terminated: {e}")
        if ui_queue is not None:
            try:
                ui_queue.put({"type": "error", "message": f"Audio error: {e}"}, block=True, timeout=5.0)
            except Exception:
                pass
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()
