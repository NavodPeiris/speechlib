"""
AT: enhance_audio escribe artifacts_dir/enhanced.wav y cachea en segunda llamada.
"""
import pytest
from conftest import make_tone_wav
from speechlib.audio_state import AudioState
from speechlib.resample_to_16k import resample_to_16k
from speechlib.loudnorm import loudnorm
from speechlib.enhance_audio import enhance_audio

pytestmark = pytest.mark.slow


def _prep_state(tmp_path):
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.3, duration_s=2.0)
    state = AudioState(source_path=wav, working_path=wav)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    state = resample_to_16k(state)
    state = loudnorm(state)
    return state


def test_enhance_writes_to_artifacts_dir(tmp_path):
    """enhance_audio produce artifacts_dir/enhanced.wav."""
    state = _prep_state(tmp_path)
    result = enhance_audio(state)

    assert result.is_enhanced is True
    assert result.working_path == result.artifacts_dir / "enhanced.wav"
    assert result.working_path.exists()


def test_enhance_cache_skips_on_second_call(tmp_path):
    """Segunda llamada a enhance_audio no regenera el archivo (cache hit)."""
    state = _prep_state(tmp_path)
    enhance_audio(state)

    enhanced = state.artifacts_dir / "enhanced.wav"
    mtime_first = enhanced.stat().st_mtime

    enhance_audio(state)
    mtime_second = enhanced.stat().st_mtime

    assert mtime_second == mtime_first, (
        f"enhanced.wav fue regenerado: mtime cambio de {mtime_first} a {mtime_second}"
    )
