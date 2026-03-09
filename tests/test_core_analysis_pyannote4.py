"""
Unit tests: pyannote 4.x API migration.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
from conftest import make_wav


def _make_mock_pipeline(mock_cls, segments):
    """Helper: configura mock_cls.from_pretrained para devolver pipeline con segments."""
    mock_pipeline = MagicMock()
    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = segments
    mock_pipeline.return_value = mock_diarization
    mock_cls.from_pretrained.return_value = mock_pipeline
    return mock_pipeline, mock_diarization


def test_from_pretrained_uses_community1_model(tmp_path):
    """Verifica que se usa el modelo community-1 y token= en vez de use_auth_token=."""
    wav = make_wav(tmp_path / "audio.wav", n_frames=1600)

    with (
        patch("speechlib.core_analysis.Pipeline") as mock_cls,
        patch("speechlib.core_analysis.torchaudio") as mock_torchaudio,
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.wav_file_segmentation", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
    ):
        mock_torchaudio.load.return_value = (MagicMock(), 16000)
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        _make_mock_pipeline(mock_cls, [(mock_seg, None, "SPEAKER_00")])

        from speechlib.core_analysis import core_analysis

        core_analysis(str(wav), None, "logs", "en", "tiny", "MY_TOKEN", "whisper")

        args, kwargs = mock_cls.from_pretrained.call_args
        assert args[0] == "pyannote/speaker-diarization-community-1"
        assert "use_auth_token" not in kwargs
        assert kwargs.get("token") == "MY_TOKEN"


def test_pipeline_called_with_file_path(tmp_path):
    """Verifica que el pipeline se llama con un string de path, no con dict."""
    wav = make_wav(tmp_path / "audio.wav", n_frames=1600)

    with (
        patch("speechlib.core_analysis.Pipeline") as mock_cls,
        patch("speechlib.core_analysis.torchaudio") as mock_torchaudio,
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.wav_file_segmentation", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
    ):
        mock_torchaudio.load.return_value = (MagicMock(), 16000)
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_pipeline, _ = _make_mock_pipeline(
            mock_cls, [(mock_seg, None, "SPEAKER_00")]
        )

        from speechlib.core_analysis import core_analysis

        core_analysis(str(wav), None, "logs", "en", "tiny", "TOKEN", "whisper")

        call_args = mock_pipeline.call_args
        assert isinstance(call_args[0][0], str)
        assert "waveform" not in str(call_args)


def test_itertracks_still_yields_3tuple(tmp_path):
    """El loop de diarización sigue funcionando con (turn, _, speaker)."""
    wav = make_wav(tmp_path / "audio.wav", n_frames=1600)

    with (
        patch("speechlib.core_analysis.Pipeline") as mock_cls,
        patch("speechlib.core_analysis.torchaudio") as mock_torchaudio,
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch(
            "speechlib.core_analysis.wav_file_segmentation",
            return_value=[[0.0, 1.0, "hello"]],
        ),
        patch("speechlib.core_analysis.write_log_file"),
    ):
        mock_torchaudio.load.return_value = (MagicMock(), 16000)
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        _make_mock_pipeline(mock_cls, [(mock_seg, None, "SPEAKER_00")])

        from speechlib.core_analysis import core_analysis

        result = core_analysis(str(wav), None, "logs", "en", "tiny", "TOKEN", "whisper")

        assert any(seg[3] == "SPEAKER_00" for seg in result)
