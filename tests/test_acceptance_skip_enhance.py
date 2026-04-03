"""AT: core_analysis acepta skip_enhance=True y omite enhance_audio.

Para grabaciones de alta calidad o cuando latencia < calidad, enhance_audio
puede saltarse completamente. El pipeline sigue siendo funcional: resample →
loudnorm → diarize → transcribe (sin SE).
"""
from unittest.mock import patch, MagicMock, call
from pathlib import Path


def _make_pipeline_mocks(tmp_path):
    """Fixtures comunes para tests de core_analysis."""
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)  # fake WAV

    mock_diarization = MagicMock()
    mock_turn = MagicMock()
    mock_turn.start = 0.0
    mock_turn.end = 2.0
    mock_diarization.itertracks.return_value = [(mock_turn, None, "SPEAKER_00")]

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization

    return wav, mock_pipeline


def test_enhance_audio_not_called_when_skip_enhance(tmp_path):
    """Con skip_enhance=True, enhance_audio nunca debe invocarse."""
    from speechlib import core_analysis as ca
    wav, mock_pipeline = _make_pipeline_mocks(tmp_path)

    with (
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio") as mock_enhance,
        patch("speechlib.core_analysis._get_diarization_pipeline", return_value=mock_pipeline),
        patch("torchaudio.load", return_value=(MagicMock(), 16000)),
        patch("speechlib.core_analysis.transcribe_full_aligned", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis.merge_short_turns", side_effect=lambda s: s),
    ):
        ca.core_analysis(
            str(wav), None, str(tmp_path), "es", "large-v3-turbo",
            "fake_token", "faster-whisper", skip_enhance=True,
        )

    mock_enhance.assert_not_called()


def test_enhance_audio_called_by_default(tmp_path):
    """Sin skip_enhance (o False), enhance_audio se invoca normalmente."""
    from speechlib import core_analysis as ca
    wav, mock_pipeline = _make_pipeline_mocks(tmp_path)

    with (
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s) as mock_enhance,
        patch("speechlib.core_analysis._get_diarization_pipeline", return_value=mock_pipeline),
        patch("torchaudio.load", return_value=(MagicMock(), 16000)),
        patch("speechlib.core_analysis.transcribe_full_aligned", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis.merge_short_turns", side_effect=lambda s: s),
    ):
        ca.core_analysis(
            str(wav), None, str(tmp_path), "es", "large-v3-turbo",
            "fake_token", "faster-whisper",
        )

    mock_enhance.assert_called_once()
