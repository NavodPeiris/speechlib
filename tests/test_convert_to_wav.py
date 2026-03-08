"""
Unit tests: convert_to_wav con AudioState (Slice 2)
"""

import shutil
from pathlib import Path
from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav
from conftest import make_wav


FIXTURES = Path(__file__).parent / "fixtures"


def test_mp3_input_converts_to_wav(tmp_path):
    mp3 = shutil.copy(FIXTURES / "sample.mp3", tmp_path / "audio.mp3")
    mp3 = Path(mp3)
    original = mp3.read_bytes()
    state = AudioState(source_path=mp3, working_path=mp3)
    result = convert_to_wav(state)
    assert result.working_path.suffix == ".wav"
    assert result.working_path != mp3
    assert result.working_path.exists()
    assert result.is_wav is True
    assert mp3.read_bytes() == original


def test_wav_input_sets_is_wav_true(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    state = AudioState(source_path=wav, working_path=wav)
    result = convert_to_wav(state)
    assert result.is_wav is True


def test_wav_input_keeps_same_working_path(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    state = AudioState(source_path=wav, working_path=wav)
    result = convert_to_wav(state)
    assert result.working_path == wav


def test_wav_input_source_unchanged(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    original = wav.read_bytes()
    state = AudioState(source_path=wav, working_path=wav)
    convert_to_wav(state)
    assert wav.read_bytes() == original


def test_returns_audio_state(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    state = AudioState(source_path=wav, working_path=wav)
    result = convert_to_wav(state)
    assert isinstance(result, AudioState)


def test_source_path_never_changes(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    state = AudioState(source_path=wav, working_path=wav)
    result = convert_to_wav(state)
    assert result.source_path == wav
