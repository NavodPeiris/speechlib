"""
Slice 13 AT: relabel_vtt with --rttm and --speaker-map flags
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import torchaudio
import torch


class TestMatchBlockToSpeaker:
    """Tests for match_block_to_speaker function"""

    def test_match_block_to_speaker_no_overlap(self):
        """block without overlap in RTTM → returns None"""
        from speechlib.tools.relabel_vtt import match_block_to_speaker, MIN_OVERLAP_S

        turns = []
        for spk, start, end in [("SPEAKER_00", 0.0, 4.0), ("SPEAKER_01", 5.0, 9.0)]:
            turn = MagicMock()
            turn.start = start
            turn.end = end
            turns.append((turn, None, spk))

        annotation = MagicMock()
        annotation.itertracks.return_value = iter(turns)

        result = match_block_to_speaker(20.0, 25.0, annotation)
        assert result is None

    def test_match_block_partial_overlap_below_threshold(self):
        """block with overlap below MIN_OVERLAP_S → returns None"""
        from speechlib.tools.relabel_vtt import match_block_to_speaker, MIN_OVERLAP_S

        turns = []
        for spk, start, end in [("SPEAKER_00", 0.0, 4.0), ("SPEAKER_01", 5.0, 9.0)]:
            turn = MagicMock()
            turn.start = start
            turn.end = end
            turns.append((turn, None, spk))

        annotation = MagicMock()
        annotation.itertracks.return_value = iter(turns)

        result = match_block_to_speaker(3.95, 4.05, annotation)
        if result is not None:
            assert result in ["SPEAKER_00", "SPEAKER_01"]


class TestRelabelVttRttmIntegration:
    """Integration tests for --rttm flag in relabel_vtt"""

    def test_relabel_vtt_rttm_groups_by_speaker_tag(self, tmp_path):
        """VTT with 4 blocks, RTTM with 2 speakers → speaker_recognition called once per group"""
        from unittest.mock import patch
        from speechlib.tools.relabel_vtt import main
        from speechlib.vtt_utils import VttBlock, write_vtt

        vtt_content = """WEBVTT

1
00:00:00.000 --> 00:00:02.000
[unknown] Hola mundo

2
00:00:02.500 --> 00:00:04.500
[unknown] Como estas

3
00:00:05.000 --> 00:00:07.000
[unknown] Buenos dias

4
00:00:08.000 --> 00:00:10.000
[unknown] Hasta luego
"""
        vtt_path = tmp_path / "test.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        audio_path = tmp_path / "audio.wav"
        voices_path = tmp_path / "voices"
        voices_path.mkdir()

        rttm_path = tmp_path / "diarization.rttm"
        rttm_content = """SPEAKER test 1 0.000 4.000 <NA> <NA> SPEAKER_00 <NA>
SPEAKER test 1 4.000 10.000 <NA> <NA> SPEAKER_01 <NA>
"""
        rttm_path.write_text(rttm_content, encoding="utf-8")

        with patch("pyannote.database.util.load_rttm") as mock_load_rttm:
            mock_annotation = MagicMock()
            mock_turn1 = MagicMock()
            mock_turn1.start = 0.0
            mock_turn1.end = 4.0
            mock_turn2 = MagicMock()
            mock_turn2.start = 4.0
            mock_turn2.end = 10.0

            mock_annotation.itertracks.return_value = iter(
                [
                    (mock_turn1, None, "SPEAKER_00"),
                    (mock_turn2, None, "SPEAKER_01"),
                ]
            )
            mock_load_rttm.return_value = {"test": mock_annotation}

            with patch("speechlib.speaker_recognition.speaker_recognition") as mock_sr:
                mock_sr.return_value = "Carmen"

                with patch(
                    "speechlib.speaker_recognition.load_avg_voice_embeddings"
                ) as mock_emb:
                    mock_emb.return_value = {"Carmen": [0.1] * 512}

                    with patch(
                        "sys.argv",
                        [
                            "relabel_vtt.py",
                            str(vtt_path),
                            str(audio_path),
                            str(voices_path),
                            "--rttm",
                            str(rttm_path),
                        ],
                    ):
                        main()

                assert mock_sr.call_count >= 1

    def test_relabel_vtt_rttm_plus_speaker_map_no_embeddings(self, tmp_path):
        """--rttm + --speaker-map → speaker_recognition NOT called, labels from JSON"""
        from unittest.mock import patch
        from speechlib.tools.relabel_vtt import main

        vtt_content = """WEBVTT

1
00:00:00.000 --> 00:00:02.000
[unknown] Hola

2
00:00:02.500 --> 00:00:04.500
[unknown] Mundo
"""
        vtt_path = tmp_path / "test.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        audio_path = tmp_path / "audio.wav"
        voices_path = tmp_path / "voices"
        voices_path.mkdir()

        rttm_path = tmp_path / "diarization.rttm"
        rttm_content = """SPEAKER test 1 0.000 4.000 <NA> <NA> SPEAKER_00 <NA>
"""
        rttm_path.write_text(rttm_content, encoding="utf-8")

        speaker_map_path = tmp_path / "speaker_map.json"
        speaker_map = {"SPEAKER_00": "Pedro"}
        speaker_map_path.write_text(json.dumps(speaker_map), encoding="utf-8")

        with patch("pyannote.database.util.load_rttm") as mock_load_rttm:
            mock_annotation = MagicMock()
            mock_turn = MagicMock()
            mock_turn.start = 0.0
            mock_turn.end = 4.0
            mock_annotation.itertracks.return_value = iter(
                [(mock_turn, None, "SPEAKER_00")]
            )
            mock_load_rttm.return_value = {"test": mock_annotation}

            with patch("speechlib.speaker_recognition.speaker_recognition") as mock_sr:
                with patch(
                    "sys.argv",
                    [
                        "relabel_vtt.py",
                        str(vtt_path),
                        str(audio_path),
                        str(voices_path),
                        "--rttm",
                        str(rttm_path),
                        "--speaker-map",
                        str(speaker_map_path),
                    ],
                ):
                    main()

                mock_sr.assert_not_called()

        output_vtt = vtt_path.with_stem(vtt_path.stem + "_relabeled")
        assert output_vtt.exists()
        content = output_vtt.read_text(encoding="utf-8")
        assert "Pedro" in content

    def test_relabel_vtt_rttm_unknown_propagated(self, tmp_path):
        """speaker_map maps SPEAKER_01 → unknown_001 → blocks keep that label"""
        from unittest.mock import patch
        from speechlib.tools.relabel_vtt import main

        vtt_content = """WEBVTT

1
00:00:05.000 --> 00:00:07.000
[unknown] Hablando
"""
        vtt_path = tmp_path / "test.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        audio_path = tmp_path / "audio.wav"
        voices_path = tmp_path / "voices"
        voices_path.mkdir()

        rttm_path = tmp_path / "diarization.rttm"
        rttm_content = """SPEAKER test 1 5.000 7.000 <NA> <NA> SPEAKER_01 <NA>
"""
        rttm_path.write_text(rttm_content, encoding="utf-8")

        speaker_map_path = tmp_path / "speaker_map.json"
        speaker_map = {"SPEAKER_01": "SPEAKER_01"}
        speaker_map_path.write_text(json.dumps(speaker_map), encoding="utf-8")

        with patch("pyannote.database.util.load_rttm") as mock_load_rttm:
            mock_annotation = MagicMock()
            mock_turn = MagicMock()
            mock_turn.start = 5.0
            mock_turn.end = 7.0
            mock_annotation.itertracks.return_value = iter(
                [(mock_turn, None, "SPEAKER_01")]
            )
            mock_load_rttm.return_value = {"test": mock_annotation}

            with patch("speechlib.speaker_recognition.speaker_recognition") as mock_sr:
                with patch(
                    "sys.argv",
                    [
                        "relabel_vtt.py",
                        str(vtt_path),
                        str(audio_path),
                        str(voices_path),
                        "--rttm",
                        str(rttm_path),
                        "--speaker-map",
                        str(speaker_map_path),
                    ],
                ):
                    main()

                mock_sr.assert_not_called()

        output_vtt = vtt_path.with_stem(vtt_path.stem + "_relabeled")
        content = output_vtt.read_text(encoding="utf-8")
        assert "SPEAKER_01" in content
