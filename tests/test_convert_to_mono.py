"""
Unit tests: convert_to_mono con AudioState (Slice 3)
"""
import wave
import struct
from pathlib import Path
import pytest
from speechlib.audio_state import AudioState
from speechlib.convert_to_mono import convert_to_mono


def make_wav(path: Path, channels=2, sampwidth=2, framerate=16000, n_frames=160):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        data = struct.pack(f"<{n_frames * channels}h", *([0] * n_frames * channels))
        f.writeframes(data)
    return path


def test_stereo_creates_new_file(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    result = convert_to_mono(state)
    assert result.working_path != wav
    assert result.working_path.exists()


def test_stereo_sets_is_mono_true(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    result = convert_to_mono(state)
    assert result.is_mono is True


def test_stereo_source_not_modified(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    original = wav.read_bytes()
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    convert_to_mono(state)
    assert wav.read_bytes() == original


def test_mono_keeps_same_working_path(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=1)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    result = convert_to_mono(state)
    assert result.working_path == wav
    assert result.is_mono is True


def test_returns_audio_state(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    result = convert_to_mono(state)
    assert isinstance(result, AudioState)


def test_source_path_never_changes(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    result = convert_to_mono(state)
    assert result.source_path == wav
