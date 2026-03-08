"""
Stub de dependencias pesadas (ML/audio) para tests unitarios.
Se ejecuta antes de cualquier import de speechlib.
"""

import sys
import wave
import struct
from pathlib import Path
from unittest.mock import MagicMock

# Dependencias con modelos grandes que no deben cargarse en unit tests
_heavy = [
    "pyannote",
    "pyannote.audio",
    "torch",
    "torchaudio",
    "whisper",
    "faster_whisper",
    "transformers",
    "speechbrain",
    "speechbrain.inference",
    "assemblyai",
]

for mod in _heavy:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()


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
