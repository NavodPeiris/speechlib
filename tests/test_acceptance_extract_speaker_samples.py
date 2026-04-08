"""
Slice 4 AT (service): extract_speaker_samples ejecuta los planes contra
audio real, escribiendo WAVs en <output_dir>/<speaker_label>/clip_NN.wav.

Tests sin mocks: el audio se genera sinteticamente con torchaudio en
tmp_path. Cero fixtures, cero stubs.

GOOS-sin-mocks: el dominio (planner) ya esta probado puramente; aqui
verificamos solamente que el mutable shell escribe lo que el plan dice.
"""

from pathlib import Path

import pytest


def _write_synthetic_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> Path:
    """Genera un WAV mono con un tono senoidal — datos validos, sin dependencias externas."""
    import torch
    import torchaudio

    n = int(duration_s * sample_rate)
    t = torch.linspace(0, duration_s, n).unsqueeze(0)
    waveform = 0.1 * torch.sin(2 * torch.pi * 440 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate, bits_per_sample=16)
    return path


def test_service_writes_wavs_per_speaker_label(tmp_path):
    from speechlib.domain.sample_extraction import SampleClip, SpeakerSamplePlan
    from speechlib.services.extract_samples import extract_speaker_samples

    # 10 segundos de senoidal
    audio = _write_synthetic_wav(tmp_path / "audio.wav", duration_s=10.0)
    out_dir = tmp_path / "samples"

    plans = (
        SpeakerSamplePlan(
            speaker_label="Manuel Olguin",
            is_identified=True,
            clips=(
                SampleClip(start_ms=0, end_ms=2000),
                SampleClip(start_ms=3000, end_ms=5000),
            ),
        ),
        SpeakerSamplePlan(
            speaker_label="SPEAKER_07",
            is_identified=False,
            clips=(SampleClip(start_ms=6000, end_ms=8500),),
        ),
    )

    result = extract_speaker_samples(plans, audio, out_dir)

    # Estructura: <out_dir>/<speaker_label>/clip_NN.wav
    manuel_dir = out_dir / "Manuel Olguin"
    spk07_dir = out_dir / "SPEAKER_07"
    assert manuel_dir.is_dir()
    assert spk07_dir.is_dir()

    manuel_wavs = sorted(manuel_dir.glob("*.wav"))
    spk07_wavs = sorted(spk07_dir.glob("*.wav"))
    assert len(manuel_wavs) == 2
    assert len(spk07_wavs) == 1

    # El return refleja exactamente lo escrito en disco
    assert "Manuel Olguin" in result
    assert "SPEAKER_07" in result
    assert sorted(result["Manuel Olguin"]) == manuel_wavs
    assert sorted(result["SPEAKER_07"]) == spk07_wavs


def test_extracted_clips_have_correct_duration(tmp_path):
    """Sanity check: los WAVs cortados tienen la duracion del SampleClip."""
    import torchaudio
    from speechlib.domain.sample_extraction import SampleClip, SpeakerSamplePlan
    from speechlib.services.extract_samples import extract_speaker_samples

    audio = _write_synthetic_wav(tmp_path / "src.wav", duration_s=10.0)
    out_dir = tmp_path / "out"

    plans = (
        SpeakerSamplePlan(
            speaker_label="X",
            is_identified=True,
            clips=(SampleClip(start_ms=1000, end_ms=3500),),  # 2.5s
        ),
    )
    result = extract_speaker_samples(plans, audio, out_dir)

    wav_path = result["X"][0]
    waveform, sr = torchaudio.load(str(wav_path))
    duration_s = waveform.shape[1] / sr
    assert duration_s == pytest.approx(2.5, abs=0.05)
