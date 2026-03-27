"""
Unit tests: pyannote 4.x API migration.
Consolidados para evitar redundancia de coverage.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
from conftest import make_wav
import numpy as np


def _make_mock_pipeline(mock_cls, segments):
    mock_pipeline = MagicMock()
    mock_diarization = MagicMock(spec=[])
    mock_diarization.itertracks = MagicMock(return_value=segments)
    mock_pipeline.return_value = mock_diarization
    mock_cls.from_pretrained.return_value = mock_pipeline
    return mock_pipeline, mock_diarization


def _make_mock_segment(start, end, speaker):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    return seg


def test_pyannote4_api_comprehensive(tmp_path):
    """Verifica modelo 3.1, token=, dict {waveform, sample_rate}, y itertracks 3-tuple."""
    wav = make_wav(tmp_path / "audio.wav", n_frames=1600)

    with (
        patch("speechlib.diarization.Pipeline") as mock_cls,
        patch("speechlib.core_analysis.torchaudio") as mock_torchaudio,
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s),
        patch(
            "speechlib.core_analysis.wav_file_segmentation",
            return_value=[[0.0, 1.0, "hello", "SPEAKER_00"]],
        ),
        patch("speechlib.core_analysis.write_log_file"),
    ):
        mock_torchaudio.load.return_value = (MagicMock(), 16000)
        mock_seg = _make_mock_segment(0.0, 1.0, "SPEAKER_00")
        mock_pipeline, _ = _make_mock_pipeline(
            mock_cls, [(mock_seg, None, "SPEAKER_00")]
        )

        from speechlib.core_analysis import core_analysis

        result = core_analysis(
            str(wav), None, "logs", "en", "tiny", "MY_TOKEN", "whisper"
        )

        args, kwargs = mock_cls.from_pretrained.call_args
        assert args[0] == "pyannote/speaker-diarization-3.1", "Debe usar modelo 3.1"
        assert "use_auth_token" not in kwargs, "No debe usar use_auth_token="
        assert kwargs.get("token") == "MY_TOKEN", "Debe usar token="

        call_args = mock_pipeline.call_args
        passed = call_args[0][0]
        assert isinstance(passed, dict), f"Se esperaba dict, se pasó {type(passed)}"
        assert "waveform" in passed, "Debe pasar waveform"
        assert "sample_rate" in passed, "Debe pasar sample_rate"

        assert any(seg[3] == "SPEAKER_00" for seg in result), (
            "itertracks debe devolver 3-tuplas"
        )
