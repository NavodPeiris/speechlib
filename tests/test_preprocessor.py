"""
Unit tests: PreProcessor public API.
"""

from pathlib import Path
from speechlib.speechlib import PreProcessor
from conftest import make_wav


def test_convert_to_wav_returns_str(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    prep = PreProcessor()
    result = prep.convert_to_wav(str(wav))
    assert isinstance(result, str)
    assert result.endswith(".wav")


def test_convert_to_mono_returns_str(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    prep = PreProcessor()
    result = prep.convert_to_mono(str(wav))
    assert isinstance(result, str)
    assert Path(result).exists()


def test_re_encode_returns_str(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    prep = PreProcessor()
    result = prep.re_encode(str(wav))
    assert isinstance(result, str)
    assert Path(result).exists()


def test_convert_to_wav_source_unchanged(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    original = wav.read_bytes()
    PreProcessor().convert_to_wav(str(wav))
    assert wav.read_bytes() == original


def test_convert_to_mono_source_unchanged(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    original = wav.read_bytes()
    PreProcessor().convert_to_mono(str(wav))
    assert wav.read_bytes() == original


def test_re_encode_source_unchanged(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    original = wav.read_bytes()
    PreProcessor().re_encode(str(wav))
    assert wav.read_bytes() == original
