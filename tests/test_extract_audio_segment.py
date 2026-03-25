"""AT: audio_utils.extract_audio_segment — torchaudio, sin ffmpeg."""
import wave
from pathlib import Path
import torch
import torchaudio
import pytest


def _make_stereo_wav(path: Path, sr: int = 44100, duration_s: float = 3.0):
    """Genera WAV stereo sintético."""
    n_samples = int(sr * duration_s)
    waveform = torch.zeros(2, n_samples)
    torchaudio.save(str(path), waveform, sr, bits_per_sample=16)


def test_produces_mono_16k_wav(tmp_path):
    from speechlib.audio_utils import extract_audio_segment

    src = tmp_path / "stereo.wav"
    _make_stereo_wav(src, sr=44100, duration_s=3.0)

    dest = tmp_path / "out.wav"
    extract_audio_segment(src, dest, start_s=0.0, duration_s=2.0)

    wf, sr = torchaudio.load(str(dest))
    assert sr == 16000
    assert wf.shape[0] == 1  # mono


def test_slice_matches_time_window(tmp_path):
    from speechlib.audio_utils import extract_audio_segment

    src = tmp_path / "src.wav"
    _make_stereo_wav(src, sr=16000, duration_s=5.0)

    dest = tmp_path / "slice.wav"
    extract_audio_segment(src, dest, start_s=1.0, duration_s=2.0)

    wf, sr = torchaudio.load(str(dest))
    actual_duration = wf.shape[1] / sr
    assert abs(actual_duration - 2.0) < 0.01


def test_returns_dest_path(tmp_path):
    from speechlib.audio_utils import extract_audio_segment

    src = tmp_path / "src.wav"
    _make_stereo_wav(src, sr=16000, duration_s=2.0)
    dest = tmp_path / "sub" / "out.wav"

    result = extract_audio_segment(src, dest, start_s=0.0, duration_s=1.0)
    assert result == dest
    assert dest.exists()


def test_creates_parent_dir(tmp_path):
    from speechlib.audio_utils import extract_audio_segment

    src = tmp_path / "src.wav"
    _make_stereo_wav(src, sr=16000, duration_s=2.0)
    dest = tmp_path / "nested" / "deep" / "out.wav"

    extract_audio_segment(src, dest, start_s=0.0, duration_s=1.0)
    assert dest.exists()
