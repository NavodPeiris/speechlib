"""
Unit tests: AudioState model (Slice 1)
"""
import pytest
from pathlib import Path
from speechlib.audio_state import AudioState


def test_audio_state_defaults():
    state = AudioState(source_path=Path("a.wav"), working_path=Path("a.wav"))
    assert state.source_path == Path("a.wav")
    assert state.working_path == Path("a.wav")
    assert state.is_wav is False
    assert state.is_mono is False
    assert state.is_16bit is False


def test_audio_state_source_is_immutable():
    state = AudioState(source_path=Path("a.wav"), working_path=Path("a.wav"))
    with pytest.raises(Exception):
        state.source_path = Path("b.wav")


def test_audio_state_model_copy_updates_working_path():
    state = AudioState(source_path=Path("a.wav"), working_path=Path("a.wav"))
    updated = state.model_copy(update={"working_path": Path("b.wav"), "is_wav": True})
    assert updated.source_path == Path("a.wav")   # source sin cambios
    assert updated.working_path == Path("b.wav")
    assert updated.is_wav is True


def test_audio_state_accepts_flags():
    state = AudioState(
        source_path=Path("a.wav"),
        working_path=Path("b.wav"),
        is_wav=True,
        is_mono=True,
        is_16bit=True,
    )
    assert state.is_wav and state.is_mono and state.is_16bit
