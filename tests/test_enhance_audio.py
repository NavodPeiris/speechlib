"""Unit tests: speechlib/enhance_audio.py

Ejecutan el modelo real MossFormer2_SE_48K (lentos, ~30s primera vez).
"""
import pytest
import torchaudio
from pathlib import Path
from conftest import make_tone_wav
from speechlib.enhance_audio import enhance_audio
from speechlib.audio_state import AudioState

pytestmark = pytest.mark.slow


def _state(path: Path) -> AudioState:
    return AudioState(source_path=path, working_path=path,
                      is_wav=True, is_mono=True, is_16bit=True, is_16khz=True, is_normalized=True)


def test_enhance_audio_creates_enhanced_file(tmp_path):
    """enhance_audio crea un archivo con sufijo _enhanced."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.3, duration_s=2.0)
    result = enhance_audio(_state(wav))

    assert result.is_enhanced is True
    assert "_enhanced" in str(result.working_path)
    assert result.working_path.exists()


def test_source_path_unchanged(tmp_path):
    """source_path nunca se modifica."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.3, duration_s=2.0)
    state = _state(wav)
    result = enhance_audio(state)

    assert result.source_path == state.source_path


def test_working_path_updated(tmp_path):
    """working_path apunta al archivo enhanced, no al original."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.3, duration_s=2.0)
    state = _state(wav)
    result = enhance_audio(state)

    assert result.working_path != state.working_path


def test_output_is_valid_wav(tmp_path):
    """El archivo enhanced es un WAV válido con contenido de audio."""
    import soundfile as sf
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.3, duration_s=2.0)
    result = enhance_audio(_state(wav))

    data, sr = sf.read(str(result.working_path))
    assert len(data) > 0
    assert sr > 0
