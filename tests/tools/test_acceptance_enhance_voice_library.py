"""
AT: enhance_voice_library crea _enhanced/ con WAVs procesados por MossFormer2.

Dado un voices_folder con speakers y WAVs raw,
enhance_voice_library genera _enhanced/ en cada speaker con versiones enhanced.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import torchaudio
import torch


def _make_wav(path: Path, duration_s: float = 2.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def test_enhance_voice_library_creates_enhanced_subdir(tmp_path):
    """Cada speaker obtiene _enhanced/ con WAVs procesados."""
    from speechlib.tools.enhance_voice_library import enhance_voice_library

    alice = tmp_path / "alice"
    alice.mkdir()
    _make_wav(alice / "sample1.wav")
    _make_wav(alice / "sample2.wav")

    bob = tmp_path / "bob"
    bob.mkdir()
    _make_wav(bob / "voice.wav")

    def fake_enhance(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        return _make_wav(dst)

    with patch(
        "speechlib.tools.enhance_voice_library.enhance_wav",
        side_effect=fake_enhance,
    ):
        enhance_voice_library(tmp_path)

    assert (alice / "_enhanced" / "sample1.wav").exists()
    assert (alice / "_enhanced" / "sample2.wav").exists()
    assert (bob / "_enhanced" / "voice.wav").exists()


def test_enhance_voice_library_skips_underscore_dirs(tmp_path):
    """Directorios con prefijo _ son ignorados."""
    from speechlib.tools.enhance_voice_library import enhance_voice_library

    hidden = tmp_path / "_unknown"
    hidden.mkdir()
    _make_wav(hidden / "sample.wav")

    alice = tmp_path / "alice"
    alice.mkdir()
    _make_wav(alice / "voice.wav")

    def fake_enhance(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        return _make_wav(dst)

    with patch(
        "speechlib.tools.enhance_voice_library.enhance_wav",
        side_effect=fake_enhance,
    ):
        enhance_voice_library(tmp_path)

    assert not (hidden / "_enhanced").exists()
    assert (alice / "_enhanced" / "voice.wav").exists()


def test_enhance_voice_library_skips_already_enhanced(tmp_path):
    """Si _enhanced/file.wav ya existe, no lo reprocesa."""
    from speechlib.tools.enhance_voice_library import enhance_voice_library

    alice = tmp_path / "alice"
    alice.mkdir()
    _make_wav(alice / "sample.wav")
    enh = alice / "_enhanced"
    enh.mkdir()
    _make_wav(enh / "sample.wav")  # ya existe

    with patch(
        "speechlib.tools.enhance_voice_library.enhance_wav",
    ) as mock_enhance:
        enhance_voice_library(tmp_path)

    mock_enhance.assert_not_called()
