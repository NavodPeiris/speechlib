"""AT: el pipeline de preprocessing normaliza el volumen a -14 LUFS (Slice B)."""
import torchaudio
from conftest import make_tone_wav
from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav
from speechlib.convert_to_mono import convert_to_mono
from speechlib.re_encode import re_encode
from speechlib.resample_to_16k import resample_to_16k
from speechlib.loudnorm import loudnorm

TARGET_LUFS = -14.0


def test_preprocessing_pipeline_normalizes_loud(tmp_path):
    """AT: tono a amplitud baja llega normalizado a ~-14 LUFS tras el pipeline."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.05, duration_s=2.0)

    state = AudioState(source_path=wav, working_path=wav)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)
    state = loudnorm(state)

    assert state.is_normalized is True
    waveform, sr = torchaudio.load(str(state.working_path))
    measured = torchaudio.functional.loudness(waveform, sr).item()
    assert abs(measured - TARGET_LUFS) < 1.0


def test_preprocessing_pipeline_loudnorm_idempotent(tmp_path):
    """AT: aplicar loudnorm dos veces produce el mismo nivel."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.05, duration_s=2.0)

    state = AudioState(source_path=wav, working_path=wav)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)
    state = loudnorm(state)
    state = loudnorm(state)

    assert state.is_normalized is True
    waveform, sr = torchaudio.load(str(state.working_path))
    measured = torchaudio.functional.loudness(waveform, sr).item()
    assert abs(measured - TARGET_LUFS) < 1.0
