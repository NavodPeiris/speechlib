"""
Slices 7-9 AT: detect_unknown_speakers
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torchaudio
import torch


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_diarization_mock(speakers: list[str], duration_s: float = 4.0):
    turns = []
    for i, spk in enumerate(speakers):
        turn = MagicMock()
        turn.start = float(i * (duration_s + 1))
        turn.end = float(i * (duration_s + 1) + duration_s)
        turns.append((turn, None, spk))

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = turns
    mock_diarization.speaker_diarization = mock_diarization
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization
    return mock_pipeline


# ── Slice 7: vacío cuando todos conocidos ────────────────────────────────────

def test_detect_unknown_speakers_returns_empty_when_all_speakers_known(tmp_path):
    """Si speaker_recognition identifica a todos, el resultado es {}."""
    from speechlib.speaker_recognition import detect_unknown_speakers

    wav = _make_wav(tmp_path / "audio.wav")
    mock_pipeline = _make_diarization_mock(["SPEAKER_00"])

    with (
        patch("speechlib.speaker_recognition.get_diarization_pipeline",
              return_value=mock_pipeline),
        patch("speechlib.speaker_recognition.speaker_recognition",
              return_value="Agustin"),
    ):
        result = detect_unknown_speakers(wav, "fake_voices", hf_token="tk")

    assert result == {}


# ── Slice 8: retorna segmentos del speaker desconocido ───────────────────────

def test_detect_unknown_speakers_returns_segments_for_unknown_speaker(tmp_path):
    """SPEAKER_01 no reconocido → aparece en el resultado con sus segmentos."""
    from speechlib.speaker_recognition import detect_unknown_speakers

    wav = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
    mock_pipeline = _make_diarization_mock(["SPEAKER_00", "SPEAKER_01"])

    def fake_recognition(file, voices, segments, wildcards):
        spk_tag = segments[0][2] if segments else "SPEAKER_00"
        return "Agustin" if spk_tag == "SPEAKER_00" else "unknown"

    with (
        patch("speechlib.speaker_recognition.get_diarization_pipeline",
              return_value=mock_pipeline),
        patch("speechlib.speaker_recognition.speaker_recognition",
              side_effect=fake_recognition),
    ):
        result = detect_unknown_speakers(wav, "fake_voices", hf_token="tk")

    assert "SPEAKER_01" in result
    assert "SPEAKER_00" not in result
    # Los segmentos tienen [start_s, end_s]
    segs = result["SPEAKER_01"]
    assert len(segs) > 0
    assert all(len(s) == 2 for s in segs)
    assert all(s[0] < s[1] for s in segs)


def test_detect_unknown_speakers_preserves_timestamps(tmp_path):
    """Los timestamps del resultado coinciden con los de la diarización."""
    from speechlib.speaker_recognition import detect_unknown_speakers

    wav = _make_wav(tmp_path / "audio.wav", duration_s=12.0)

    # diarización con timestamps conocidos
    turns = []
    for start, end, spk in [(0.0, 4.0, "SPEAKER_00"), (5.0, 9.0, "SPEAKER_01")]:
        t = MagicMock()
        t.start = start
        t.end = end
        turns.append((t, None, spk))
    mock_diar = MagicMock()
    mock_diar.itertracks.return_value = turns
    mock_diar.speaker_diarization = mock_diar
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diar

    def fake_recognition(file, voices, segments, wildcards):
        spk_tag = segments[0][2] if segments else "SPEAKER_00"
        return "Agustin" if spk_tag == "SPEAKER_00" else "unknown"

    with (
        patch("speechlib.speaker_recognition.get_diarization_pipeline",
              return_value=mock_pipeline),
        patch("speechlib.speaker_recognition.speaker_recognition",
              side_effect=fake_recognition),
    ):
        result = detect_unknown_speakers(wav, "fake_voices", hf_token="tk")

    assert "SPEAKER_01" in result
    segs = result["SPEAKER_01"]
    assert [5.0, 9.0] in segs


def test_detect_unknown_speakers_multiple_unknowns(tmp_path):
    """Dos speakers desconocidos: ambos aparecen en el resultado."""
    from speechlib.speaker_recognition import detect_unknown_speakers

    wav = _make_wav(tmp_path / "audio.wav", duration_s=15.0)
    mock_pipeline = _make_diarization_mock(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    )

    with (
        patch("speechlib.speaker_recognition.get_diarization_pipeline",
              return_value=mock_pipeline),
        patch("speechlib.speaker_recognition.speaker_recognition",
              return_value="unknown"),
    ):
        result = detect_unknown_speakers(wav, "fake_voices", hf_token="tk")

    assert set(result.keys()) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}


# ── Slice 9: AT de integración detect → extract ──────────────────────────────

def test_detect_and_extract_creates_clips_for_unknown(tmp_path):
    """Encadenamiento completo: detect_unknown_speakers → extract_unknown_speakers."""
    from speechlib.speaker_recognition import detect_unknown_speakers
    from speechlib.tools.extract_unknown_speakers import extract_unknown_speakers

    wav = _make_wav(tmp_path / "audio.wav", duration_s=12.0)
    output_dir = tmp_path / "unknown_speakers"
    mock_pipeline = _make_diarization_mock(["SPEAKER_00", "SPEAKER_01"])

    def fake_recognition(file, voices, segments, wildcards):
        spk_tag = segments[0][2] if segments else "SPEAKER_00"
        return "Agustin" if spk_tag == "SPEAKER_00" else "unknown"

    with (
        patch("speechlib.speaker_recognition.get_diarization_pipeline",
              return_value=mock_pipeline),
        patch("speechlib.speaker_recognition.speaker_recognition",
              side_effect=fake_recognition),
    ):
        unknown_segments = detect_unknown_speakers(wav, "fake_voices", hf_token="tk")

    assert unknown_segments  # al menos un unknown

    clips = extract_unknown_speakers(wav, unknown_segments, output_dir)

    assert clips  # se crearon carpetas
    for tag, folder in clips.items():
        assert folder.exists()
        wavs = list(folder.glob("*.wav"))
        assert len(wavs) >= 1, f"Sin clips en {folder}"
