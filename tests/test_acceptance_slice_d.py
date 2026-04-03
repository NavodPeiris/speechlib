"""AT: enhance_audio produce audio mejorado en el pipeline de preprocessing (Slice D).

Estos tests ejecutan el modelo real MossFormer2_SE_48K.
Son lentos (~30s la primera vez por carga del modelo).
"""
import pytest
import torchaudio
from conftest import make_tone_wav
from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav
from speechlib.convert_to_mono import convert_to_mono
from speechlib.re_encode import re_encode
from speechlib.resample_to_16k import resample_to_16k
from speechlib.loudnorm import loudnorm
from speechlib.enhance_audio import enhance_audio

pytestmark = pytest.mark.slow


def test_pipeline_produces_enhanced_file(tmp_path):
    """AT: el pipeline A→B→D genera un archivo _enhanced con is_enhanced=True."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.3, duration_s=2.0)

    state = AudioState(source_path=wav, working_path=wav)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)
    state = loudnorm(state)
    state = enhance_audio(state)

    assert state.is_enhanced is True
    assert state.working_path == state.artifacts_dir / "enhanced.wav"
    assert state.working_path.exists()
    assert state.source_path == wav


def test_enhanced_output_is_valid_audio(tmp_path):
    """AT: el archivo enhanced es un WAV válido con contenido de audio."""
    import soundfile as sf
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.3, duration_s=2.0)

    state = AudioState(source_path=wav, working_path=wav)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)
    state = loudnorm(state)
    state = enhance_audio(state)

    data, sr = sf.read(str(state.working_path))
    assert sr == 16000  # model preserves input sample rate
    assert len(data) > 0
