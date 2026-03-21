import wave
import struct
import math
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


def make_tone_wav(path: Path, freq=1000, amplitude=0.5, framerate=16000, duration_s=1.0):
    """WAV con tono senoidal puro — tiene contenido frecuencial real, medible en LUFS."""
    n_frames = int(framerate * duration_s)
    max_val = int(amplitude * 32767)
    samples = [
        int(max_val * math.sin(2 * math.pi * freq * i / framerate))
        for i in range(n_frames)
    ]
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        f.writeframes(struct.pack(f"<{n_frames}h", *samples))
    return path
