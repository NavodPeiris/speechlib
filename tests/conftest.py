"""
Stub de dependencias pesadas (ML/audio) para tests unitarios.
Se ejecuta antes de cualquier import de speechlib.
"""
import sys
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
    "pydub",
    "pydub.AudioSegment",
]

for mod in _heavy:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
