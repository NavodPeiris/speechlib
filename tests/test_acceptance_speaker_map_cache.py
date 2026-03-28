"""
Slice 12 AT: speaker_map.json cache
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
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


class TestSpeakerMapCache:
    """Tests for speaker_map.json caching"""

    def test_speaker_map_saved_after_recognition(self, tmp_path):
        """core_analysis saves speaker_map.json after running speaker recognition"""
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

        speaker_map_path = tmp_path / ".audio" / "speaker_map.json"
        assert speaker_map_path.exists(), "speaker_map.json should be saved"

    def test_speaker_map_cache_skips_recognition(self, tmp_path):
        """speaker_map.json exists → speaker_recognition NOT called"""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        rttm_path = tmp_path / ".audio" / "diarization.rttm"
        rttm_path.parent.mkdir(parents=True)
        rttm_path.write_text(
            "SPEAKER test 1 0.0 5.0 <NA> <NA> SPEAKER_00 <NA> <NA>", encoding="utf-8"
        )

        speaker_map_path = tmp_path / ".audio" / "speaker_map.json"
        speaker_map_path.write_text(
            json.dumps({"SPEAKER_00": "speaker"}), encoding="utf-8"
        )

        mock_annotation = _make_annotation_mock(["SPEAKER_00"])
        mock_rttm = MagicMock(return_value={"test": mock_annotation})
        mock_speaker_rec = MagicMock(return_value="speaker")

        with patch("speechlib.core_analysis._load_rttm", mock_rttm):
            with patch("speechlib.core_analysis.speaker_recognition", mock_speaker_rec):
                core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        mock_speaker_rec.assert_not_called()

    def test_speaker_map_cache_skips_diarization_and_recognition(self, tmp_path):
        """diarization.rttm + speaker_map.json exist → neither pipeline nor speaker_recognition called"""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        rttm_path = tmp_path / ".audio" / "diarization.rttm"
        rttm_path.parent.mkdir(parents=True)
        rttm_path.write_text(
            "SPEAKER test 1 0.0 5.0 <NA> <NA> SPEAKER_00 <NA> <NA>", encoding="utf-8"
        )

        speaker_map_path = tmp_path / ".audio" / "speaker_map.json"
        speaker_map_path.write_text(
            json.dumps({"SPEAKER_00": "speaker"}), encoding="utf-8"
        )

        mock_annotation = _make_annotation_mock(["SPEAKER_00"])
        mock_rttm = MagicMock(return_value={"test": mock_annotation})
        mock_pipeline = MagicMock()
        mock_speaker_rec = MagicMock(return_value="speaker")

        with patch("speechlib.core_analysis._load_rttm", mock_rttm):
            with patch(
                "speechlib.core_analysis._get_diarization_pipeline", mock_pipeline
            ):
                with patch(
                    "speechlib.core_analysis.speaker_recognition", mock_speaker_rec
                ):
                    core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        mock_pipeline.assert_not_called()
        mock_speaker_rec.assert_not_called()

    def test_speaker_map_json_format(self, tmp_path):
        """JSON has SPEAKER_XX keys with name or unknown_NNN values"""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        mock_pipeline = MagicMock()
        mock_diar = MagicMock()
        mock_diar.speaker_diarization = _make_annotation_mock(
            ["SPEAKER_00", "SPEAKER_01"]
        )
        mock_pipeline.return_value = mock_diar

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            with patch(
                "speechlib.core_analysis._load_rttm", side_effect=FileNotFoundError()
            ):
                with patch(
                    "speechlib.core_analysis.speaker_recognition",
                    return_value="unknown",
                ):
                    core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        speaker_map_path = tmp_path / ".audio" / "speaker_map.json"
        data = json.loads(speaker_map_path.read_text(encoding="utf-8"))

        assert "SPEAKER_00" in data
        assert "SPEAKER_01" in data
        assert data["SPEAKER_00"].startswith("SPEAKER_")
        assert data["SPEAKER_01"].startswith("SPEAKER_")
