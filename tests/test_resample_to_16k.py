"""Unit tests: speechlib/resample_to_16k.py"""
import torchaudio
from pathlib import Path
from conftest import make_wav
from speechlib.resample_to_16k import resample_to_16k
from speechlib.audio_state import AudioState


def _state(path: Path) -> AudioState:
    return AudioState(source_path=path, working_path=path,
                      is_wav=True, is_mono=True, is_16bit=True)


def test_non_16k_audio_is_resampled(tmp_path):
    """44100 Hz → nuevo archivo _16k.wav a 16000 Hz."""
    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)
    result = resample_to_16k(_state(wav))

    assert result.is_16khz is True
    assert result.working_path != wav
    assert "_16k" in result.working_path.name
    assert result.working_path.exists()
    _, sr = torchaudio.load(str(result.working_path))
    assert sr == 16000


def test_already_16k_passes_through(tmp_path):
    """16000 Hz → mismo working_path, sin crear archivo _16k."""
    wav = make_wav(tmp_path / "audio.wav", framerate=16000, n_frames=1600)
    result = resample_to_16k(_state(wav))

    assert result.is_16khz is True
    assert result.working_path == wav
    assert not (tmp_path / "audio_16k.wav").exists()


def test_48k_audio_is_resampled(tmp_path):
    """48000 Hz (micrófono de reunión) → 16000 Hz."""
    wav = make_wav(tmp_path / "audio.wav", framerate=48000, n_frames=4800)
    result = resample_to_16k(_state(wav))

    assert result.is_16khz is True
    _, sr = torchaudio.load(str(result.working_path))
    assert sr == 16000


def test_source_path_is_never_modified(tmp_path):
    """source_path permanece igual en todos los casos."""
    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)
    state = _state(wav)
    result = resample_to_16k(state)

    assert result.source_path == state.source_path


def test_output_is_mono(tmp_path):
    """El archivo de salida tiene 1 canal."""
    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)
    result = resample_to_16k(_state(wav))

    waveform, _ = torchaudio.load(str(result.working_path))
    assert waveform.shape[0] == 1
