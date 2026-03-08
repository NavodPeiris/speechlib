"""
Unit tests: convert_to_wav con AudioState (Slice 2)
"""
import wave
import struct
from pathlib import Path
import pytest
from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav


def make_wav(path: Path, channels=1, sampwidth=2, framerate=16000, n_frames=160):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        data = struct.pack(f"<{n_frames * channels}h", *([0] * n_frames * channels))
        f.writeframes(data)
    return path


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
