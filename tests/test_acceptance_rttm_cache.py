"""
Slice 11 AT: diarization.rttm cache
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import torchaudio
import torch


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_annotation_mock(speakers: list[str], duration_s: float = 5.0):
    turns = []
    for i, spk in enumerate(speakers):
        turn = MagicMock()
        turn.start = float(i * (duration_s + 1))
        turn.end = float(i * (duration_s + 1) + duration_s)
        turns.append((turn, None, spk))

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = iter(turns)

    def write_rttm(file_handle):
        for spk in speakers:
            start = 0.0
            file_handle.write(
                f"SPEAKER test 1 {start} {duration_s} <NA> <NA> {spk} <NA> <NA>\n"
            )

    mock_annotation.write_rttm = write_rttm
    return mock_annotation


class TestRttmCache:
    """Tests for diarization.rttm caching"""

    def test_rttm_saved_after_diarization(self, tmp_path):
        """core_analysis saves diarization.rttm after running diarization"""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        mock_pipeline = MagicMock()
        mock_diar = MagicMock()
        mock_diar.speaker_diarization = _make_annotation_mock(["SPEAKER_00"])
        mock_pipeline.return_value = mock_diar

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            with patch(
                "speechlib.core_analysis._load_rttm", side_effect=FileNotFoundError()
            ):
                core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        rttm_path = tmp_path / ".audio" / "diarization.rttm"
        assert rttm_path.exists(), f"diarization.rttm should be saved at {rttm_path}"

    def test_rttm_cache_skips_pipeline_call(self, tmp_path):
        """diarization.rttm exists → pipeline NOT called"""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        rttm_path = tmp_path / ".audio" / "diarization.rttm"
        rttm_path.parent.mkdir(parents=True)
        rttm_path.write_text(
            "SPEAKER test 1 0.0 5.0 <NA> <NA> speaker_0 <NA> <NA>", encoding="utf-8"
        )

        mock_annotation = _make_annotation_mock(["SPEAKER_00"])
        mock_rttm = MagicMock(return_value={"test": mock_annotation})

        mock_pipeline = MagicMock()

        with patch("speechlib.core_analysis._load_rttm", mock_rttm):
            with patch(
                "speechlib.core_analysis._get_diarization_pipeline",
                return_value=mock_pipeline,
            ):
                core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        mock_pipeline.assert_not_called()

    def test_rttm_content_roundtrip(self, tmp_path):
        """save annotation → load_rttm → same SPEAKER_XX and timestamps"""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        speakers_input = ["SPEAKER_00", "SPEAKER_01"]
        mock_pipeline = MagicMock()
        mock_diar = MagicMock()
        mock_diar.speaker_diarization = _make_annotation_mock(
            speakers_input, duration_s=4.0
        )
        mock_pipeline.return_value = mock_diar

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            with patch(
                "speechlib.core_analysis._load_rttm", side_effect=FileNotFoundError()
            ):
                core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        rttm_path = tmp_path / ".audio" / "diarization.rttm"
        content = rttm_path.read_text(encoding="utf-8")

        assert "SPEAKER_00" in content
        assert "SPEAKER_01" in content
