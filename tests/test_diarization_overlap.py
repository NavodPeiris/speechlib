"""Tests: core_analysis preserva segmentos solapados devueltos por pyannote 3.1."""
import pytest
from unittest.mock import patch, MagicMock
from speechlib.core_analysis import core_analysis


@pytest.fixture
def fake_wav(tmp_path):
    f = tmp_path / "audio.wav"
    f.write_bytes(b"RIFF")
    return f


def _state(path):
    s = MagicMock()
    s.working_path = path
    return s


def _make_pipeline_with_overlap():
    """Pipeline que devuelve SPEAKER_00 [0.0-5.0] y SPEAKER_01 [3.0-8.0] solapados."""
    turn_a = MagicMock(); turn_a.start = 0.0; turn_a.end = 5.0
    turn_b = MagicMock(); turn_b.start = 3.0; turn_b.end = 8.0

    mock_annotation = MagicMock(spec=[])
    mock_annotation.itertracks = MagicMock(return_value=[
        (turn_a, None, "SPEAKER_00"),
        (turn_b, None, "SPEAKER_01"),
    ])

    pipeline_instance = MagicMock()
    pipeline_instance.return_value = mock_annotation
    pipeline_instance.to = MagicMock()
    return pipeline_instance


def test_overlapping_segments_both_appear_in_result(fake_wav, tmp_path):
    """Solapamiento SPEAKER_00[0-5] / SPEAKER_01[3-8]: ambos deben estar en el resultado."""
    pipeline_mock = _make_pipeline_with_overlap()

    def fake_segmentation(file, segments, *args, **kwargs):
        return [[s[0], s[1], "text"] for s in segments]

    def fake_full_aligned(file, segments, *args, **kwargs):
        return [[s[0], s[1], "text", s[2]] for s in segments]

    with (
        patch("speechlib.core_analysis.Pipeline.from_pretrained", return_value=pipeline_mock),
        patch("speechlib.core_analysis.AudioState", return_value=_state(fake_wav)),
        patch("speechlib.core_analysis.torchaudio") as mock_torchaudio,
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s),
        patch("speechlib.core_analysis.wav_file_segmentation", side_effect=fake_segmentation),
        patch("speechlib.core_analysis.transcribe_full_aligned", side_effect=fake_full_aligned),
        patch("speechlib.core_analysis.write_log_file"),
    ):
        mock_torchaudio.load.return_value = (MagicMock(), 16000)
        result = core_analysis(
            str(fake_wav), None, str(tmp_path), "en", "base", "hf-token", "faster-whisper"
        )

    speakers_out = {seg[3] for seg in result}
    assert "SPEAKER_00" in speakers_out, "SPEAKER_00 desapareció del resultado"
    assert "SPEAKER_01" in speakers_out, "SPEAKER_01 desapareció del resultado"


def test_overlapping_timestamps_preserved(fake_wav, tmp_path):
    """Los timestamps solapados no deben modificarse al procesar."""
    pipeline_mock = _make_pipeline_with_overlap()

    def fake_segmentation(file, segments, *args, **kwargs):
        return [[s[0], s[1], "text"] for s in segments]

    def fake_full_aligned(file, segments, *args, **kwargs):
        return [[s[0], s[1], "text", s[2]] for s in segments]

    with (
        patch("speechlib.core_analysis.Pipeline.from_pretrained", return_value=pipeline_mock),
        patch("speechlib.core_analysis.AudioState", return_value=_state(fake_wav)),
        patch("speechlib.core_analysis.torchaudio") as mock_torchaudio,
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s),
        patch("speechlib.core_analysis.wav_file_segmentation", side_effect=fake_segmentation),
        patch("speechlib.core_analysis.transcribe_full_aligned", side_effect=fake_full_aligned),
        patch("speechlib.core_analysis.write_log_file"),
    ):
        mock_torchaudio.load.return_value = (MagicMock(), 16000)
        result = core_analysis(
            str(fake_wav), None, str(tmp_path), "en", "base", "hf-token", "faster-whisper"
        )

    starts = {seg[0] for seg in result}
    assert 0.0 in starts, "timestamp de inicio SPEAKER_00 perdido"
    assert 3.0 in starts, "timestamp de inicio solapado SPEAKER_01 perdido"
