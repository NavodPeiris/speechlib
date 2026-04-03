"""
AT: Verify core_analysis uses pyannote 4.x API.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
from conftest import make_wav


def test_pipeline_initialized_with_3_1_model(tmp_path):
    """
    core_analysis llama a Pipeline.from_pretrained con el modelo 3.1
    y el parámetro token= (no use_auth_token=).
    """
    wav = make_wav(tmp_path / "audio.wav", n_frames=1600)

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0

    mock_pipeline = MagicMock()
    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [(mock_segment, None, "SPEAKER_00")]
    mock_pipeline.return_value = mock_diarization

    with (
        patch("speechlib.diarization.Pipeline") as mock_cls,
        patch("speechlib.core_analysis.torchaudio") as mock_torchaudio,
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s),
        patch("speechlib.core_analysis.wav_file_segmentation", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
    ):
        mock_torchaudio.load.return_value = (MagicMock(), 16000)
        mock_cls.from_pretrained.return_value = mock_pipeline

        from speechlib.core_analysis import core_analysis

        core_analysis(str(wav), None, "logs", "en", "tiny", "TOKEN", "whisper")

        mock_cls.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            token="TOKEN",
        )
