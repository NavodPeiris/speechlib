import wave
import struct
from pathlib import Path


def make_wav(path: Path, channels=1, sampwidth=2, framerate=16000, n_frames=160):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        if sampwidth == 1:
            data = bytes([128] * n_frames * channels)
        else:
            data = struct.pack(f"<{n_frames * channels}h", *([0] * n_frames * channels))
        f.writeframes(data)
    return path
