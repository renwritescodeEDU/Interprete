import multiprocessing
import queue
import numpy as np
import pyaudio
import torchaudio
import torch

CHUNK = 480  # 30ms at 16000Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


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
    except IOError:
        default_input_index = -1

    for i in range(p.get_device_count()):
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
        except Exception:
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
        print(f"Failed to open audio at {RATE}Hz: {e}")
        print("Attempting to open at device default rate and will resample...")

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
        print(f"Successfully opened audio at {actual_rate}Hz. Will resample to {RATE}Hz.")
        return stream, actual_rate, buffer_size
    except Exception as e2:
        raise RuntimeError(f"Failed to open audio stream: {e2}")


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
        print(e)
        p.terminate()
        return

    frames = []
    
    # Send partial transcript every ~1 second of accumulated speech
    partial_threshold = int((1.0 * actual_rate) / buffer_size)  
    frames_since_last_partial = 0
    partial_start_index = 0
    is_recording = False

    def process_audio(frames_list):
        audio_bytes = b"".join(frames_list)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if actual_rate != RATE:
            audio_tensor = torch.from_numpy(audio_array)
            audio_tensor = torchaudio.functional.resample(audio_tensor, actual_rate, RATE)
            audio_array = audio_tensor.numpy()
            
        return audio_array

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
                    # Flush any stale audio from hardware buffer
                    try:
                        while stream.get_read_available() > 0:
                            stream.read(stream.get_read_available(), exception_on_overflow=False)
                    except Exception:
                        pass
                
                elif cmd == "FINISH" and is_recording:
                    # Clear any stale partials from the queue so the final chunk gets processed immediately
                    while not asr_queue.empty():
                        try:
                            asr_queue.get_nowait()
                        except queue.Empty:
                            break

                    if len(frames) > 0:
                        audio_array = process_audio(frames)
                        try:
                            asr_queue.put((audio_array, RATE, True), block=True, timeout=2.0)
                        except queue.Full:
                            pass
                    else:
                        try:
                            asr_queue.put((np.array([], dtype=np.float32), RATE, True), block=True, timeout=2.0)
                        except queue.Full:
                            pass
                    frames = []
                    frames_since_last_partial = 0
                    partial_start_index = 0
                    is_recording = False

                elif isinstance(cmd, tuple) and len(cmd) == 2 and cmd[0] == "SET_DEVICE":
                    # Switch audio device (only allowed when not recording)
                    if not is_recording:
                        new_device_index = cmd[1]
                        print(f"Switching audio device to index {new_device_index}...")
                        try:
                            stream.stop_stream()
                            stream.close()
                        except Exception:
                            pass
                        try:
                            stream, actual_rate, buffer_size = _open_stream(p, new_device_index)
                            partial_threshold = int((1.0 * actual_rate) / buffer_size)
                            device_index = new_device_index
                            print(f"Audio device switched successfully.")
                        except RuntimeError as e:
                            print(f"Failed to switch device: {e}. Reopening previous device...")
                            try:
                                stream, actual_rate, buffer_size = _open_stream(p, device_index)
                                partial_threshold = int((1.0 * actual_rate) / buffer_size)
                            except RuntimeError:
                                print("FATAL: Cannot reopen any audio device.")
                                break
                    else:
                        print("Cannot switch device while recording.")

            except queue.Empty:
                pass

            # Only read and process frames if actively recording
            if is_recording:
                frame = stream.read(buffer_size, exception_on_overflow=False)
                frames.append(frame)
                frames_since_last_partial += 1
                
                max_frames = int(5 * 60 * actual_rate / buffer_size)
                if len(frames) > max_frames:
                    dropped_frames = len(frames) - max_frames
                    frames = frames[-max_frames:]
                    partial_start_index = max(0, partial_start_index - dropped_frames)
                
                # Emit a partial transcript periodically
                if frames_since_last_partial >= partial_threshold and len(frames) > partial_start_index:
                    partial_frames = frames[partial_start_index:]
                    audio_array = process_audio(partial_frames)
                    try:
                        asr_queue.put((audio_array, RATE, False), block=False)
                    except queue.Full:
                        pass
                    partial_start_index = len(frames)
                    frames_since_last_partial = 0
            else:
                # Discard hardware buffer to prevent overflow while idle
                try:
                    if stream.get_read_available() >= buffer_size:
                        stream.read(stream.get_read_available(), exception_on_overflow=False)
                except Exception:
                    pass

    except Exception as e:
        print(f"Audio capture terminated: {e}")
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
