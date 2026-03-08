"""
Unit tests: re_encode con AudioState (Slice 4)
"""

import wave
import pytest
from speechlib.audio_state import AudioState
from speechlib.re_encode import re_encode
from conftest import make_wav


@pytest.fixture
def wav_8bit(tmp_path):
    return make_wav(tmp_path / "audio.wav", sampwidth=1)


@pytest.fixture
def state_8bit(wav_8bit):
    return AudioState(
        source_path=wav_8bit, working_path=wav_8bit, is_wav=True, is_mono=True
    )


@pytest.fixture
def wav_16bit(tmp_path):
    return make_wav(tmp_path / "audio.wav", sampwidth=2)


@pytest.fixture
def state_16bit(wav_16bit):
    return AudioState(
        source_path=wav_16bit, working_path=wav_16bit, is_wav=True, is_mono=True
    )


def test_8bit_creates_new_16bit_file(state_8bit):
    result = re_encode(state_8bit)
    assert result.working_path != state_8bit.working_path
    assert result.working_path.exists()


def test_8bit_sets_is_16bit_true(state_8bit):
    result = re_encode(state_8bit)
    assert result.is_16bit is True


def test_8bit_output_is_actually_16bit(state_8bit):
    result = re_encode(state_8bit)
    with wave.open(str(result.working_path), "rb") as f:
        assert f.getsampwidth() == 2


def test_already_16bit_keeps_same_working_path(state_16bit):
    result = re_encode(state_16bit)
    assert result.working_path == state_16bit.source_path
    assert result.is_16bit is True


def test_8bit_source_not_modified(state_8bit):
    original = state_8bit.source_path.read_bytes()
    re_encode(state_8bit)
    assert state_8bit.source_path.read_bytes() == original


def test_returns_audio_state(state_16bit):
    result = re_encode(state_16bit)
    assert isinstance(result, AudioState)


def test_source_path_never_changes(state_8bit):
    result = re_encode(state_8bit)
    assert result.source_path == state_8bit.source_path
