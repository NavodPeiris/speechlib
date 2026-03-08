"""
Unit tests: audio_utils slice_and_save helper.
"""

import pytest
from conftest import make_wav
from speechlib.audio_utils import slice_and_save


def test_slice_creates_file(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", n_frames=1600)
    output = tmp_path / "slice.wav"
    slice_and_save(str(wav), 0, 100, str(output))
    assert output.exists()


def test_slice_duration_is_correct(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", framerate=16000, n_frames=32000)
    output = tmp_path / "slice.wav"
    slice_and_save(str(wav), 500, 1500, str(output))
    import wave

    with wave.open(str(output), "rb") as f:
        duration = f.getnframes() / f.getframerate()
    assert 0.9 <= duration <= 1.1


def test_source_not_modified(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", n_frames=1600)
    original = wav.read_bytes()
    output = tmp_path / "slice.wav"
    slice_and_save(str(wav), 0, 100, str(output))
    assert wav.read_bytes() == original
