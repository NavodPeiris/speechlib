"""
Unit tests: re_encode con AudioState (Slice 4)
"""
import wave
import struct
from pathlib import Path
import pytest
from speechlib.audio_state import AudioState
from speechlib.re_encode import re_encode


def make_wav(path: Path, channels=1, sampwidth=2, framerate=16000, n_frames=160):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        if sampwidth == 1:
            data = bytes([128] * n_frames * channels)
        else:
            data = struct.pack(f"<{n_frames * channels}h", *([0] * n_frames * channels))
        f.writeframes(data)
    return path


def test_8bit_creates_new_16bit_file(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True, is_mono=True)
    result = re_encode(state)
    assert result.working_path != wav
    assert result.working_path.exists()


def test_8bit_sets_is_16bit_true(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True, is_mono=True)
    result = re_encode(state)
    assert result.is_16bit is True


def test_8bit_output_is_actually_16bit(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True, is_mono=True)
    result = re_encode(state)
    with wave.open(str(result.working_path), 'rb') as f:
        assert f.getsampwidth() == 2


def test_already_16bit_keeps_same_working_path(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=2)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True, is_mono=True)
    result = re_encode(state)
    assert result.working_path == wav
    assert result.is_16bit is True


def test_8bit_source_not_modified(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    original = wav.read_bytes()
    state = AudioState(source_path=wav, working_path=wav, is_wav=True, is_mono=True)
    re_encode(state)
    assert wav.read_bytes() == original


def test_returns_audio_state(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=2)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True, is_mono=True)
    result = re_encode(state)
    assert isinstance(result, AudioState)


def test_source_path_never_changes(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True, is_mono=True)
    result = re_encode(state)
    assert result.source_path == wav
