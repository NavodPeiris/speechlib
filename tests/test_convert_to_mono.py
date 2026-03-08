"""
Unit tests: convert_to_mono con AudioState (Slice 3)
"""

import pytest
from speechlib.audio_state import AudioState
from speechlib.convert_to_mono import convert_to_mono
from conftest import make_wav


@pytest.fixture
def stereo_wav(tmp_path):
    return make_wav(tmp_path / "audio.wav", channels=2)


@pytest.fixture
def stereo_state(stereo_wav):
    return AudioState(source_path=stereo_wav, working_path=stereo_wav, is_wav=True)


@pytest.fixture
def mono_wav(tmp_path):
    return make_wav(tmp_path / "audio.wav", channels=1)


def test_stereo_creates_new_file(stereo_wav):
    state = AudioState(source_path=stereo_wav, working_path=stereo_wav, is_wav=True)
    result = convert_to_mono(state)
    assert result.working_path != stereo_wav
    assert result.working_path.exists()


def test_stereo_sets_is_mono_true(stereo_state):
    result = convert_to_mono(stereo_state)
    assert result.is_mono is True


def test_stereo_source_not_modified(stereo_wav):
    original = stereo_wav.read_bytes()
    state = AudioState(source_path=stereo_wav, working_path=stereo_wav, is_wav=True)
    convert_to_mono(state)
    assert stereo_wav.read_bytes() == original


def test_mono_keeps_same_working_path(mono_wav):
    state = AudioState(source_path=mono_wav, working_path=mono_wav, is_wav=True)
    result = convert_to_mono(state)
    assert result.working_path == mono_wav
    assert result.is_mono is True


def test_returns_audio_state(stereo_state):
    result = convert_to_mono(stereo_state)
    assert isinstance(result, AudioState)


def test_source_path_never_changes(stereo_state):
    result = convert_to_mono(stereo_state)
    assert result.source_path == stereo_state.source_path
