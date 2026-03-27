"""
Slice 11 AT: propagate threshold in speaker_recognition

Uses real voice examples from examples/voices/ (obama, zach) for audio input.
Similarities:
- obama vs obama: ~0.785 (same speaker)
- obama vs zach: ~0.053 (different speakers)
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torchaudio
import torch
import numpy as np
from speechlib.speaker_recognition import SPEAKER_SIMILARITY_THRESHOLD


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


from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
VOICES_DIR = EXAMPLES_DIR / "voices"
OBAMA_VOICES = VOICES_DIR / "obama"
ZACH_VOICES = VOICES_DIR / "zach"


SPEAKER_A_EMBEDDING = np.array([1.0, 0.0, 0.0])
TEST_SIM_EMBEDDING = np.array([0.45, 0.893, 0.0])


def _cosine_sim(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


EXPECTED_SIM = _cosine_sim(SPEAKER_A_EMBEDDING, TEST_SIM_EMBEDDING)


class TestSpeakerRecognitionUsesCustomThreshold:
    """Loop interno — test unitario de speaker_recognition

    Uses real obama2.wav as input audio but mocks embeddings for controlled similarity.
    """

    @pytest.mark.parametrize(
        "threshold,expected",
        [
            (0.40, "speaker_a"),
            (0.50, "unknown"),
        ],
    )
    def test_threshold_behavior(self, tmp_path, threshold, expected):
        """Parametrized: threshold=X → expected result
        - Mock similarity = 0.45 (between 0.40 and 0.50)
        - threshold=0.40: 0.45 >= 0.40 → recognized
        - threshold=0.50: 0.45 < 0.50 → unknown
        """
        from speechlib.speaker_recognition import speaker_recognition

        wav = OBAMA_VOICES / "obama2.wav"

        voices_base = tmp_path / "voices"
        voices_base.mkdir(parents=True)
        voices_folder = voices_base / "speaker_a"
        voices_folder.mkdir(parents=True)
        _make_wav(voices_folder / "voice1.wav")

        segments = [[0.0, 4.0, "SPEAKER_00"]]

        def mock_inference(path):
            if "speaker_a" in str(path):
                return SPEAKER_A_EMBEDDING
            return TEST_SIM_EMBEDDING

        def mock_get_embedding(path):
            if "speaker_a" in str(path):
                return SPEAKER_A_EMBEDDING
            return TEST_SIM_EMBEDDING

        with (
            patch(
                "speechlib.speaker_recognition.get_embedding",
                side_effect=mock_get_embedding,
            ),
            patch("speechlib.speaker_recognition._get_inference") as mock_get_inf,
        ):
            mock_get_inf.return_value.side_effect = mock_inference
            result = speaker_recognition(
                str(wav), str(voices_base), segments, [], threshold=threshold
            )

        assert result == expected


class TestDetectUnknownSpeakersThresholdPropagated:
    """Loop externo — AT de wiring completo"""

    @pytest.mark.parametrize(
        "threshold,expected_empty",
        [
            (0.40, True),
            (0.50, False),
        ],
    )
    def test_threshold_propagation(self, tmp_path, threshold, expected_empty):
        """Parametrized: threshold=X → expected result
        - threshold=0.40: speaker recognized → empty result
        - threshold=0.50: speaker unknown → has segments
        """
        from speechlib.speaker_recognition import detect_unknown_speakers

        wav = OBAMA_VOICES / "obama2.wav"
        mock_pipeline = _make_diarization_mock(["SPEAKER_00"])

        def fake_recognition(file, voices, segments, wildcards, threshold=0.40):
            return "speaker_a" if threshold <= 0.40 else "unknown"

        with (
            patch(
                "speechlib.speaker_recognition.get_diarization_pipeline",
                return_value=mock_pipeline,
            ),
            patch(
                "speechlib.speaker_recognition.speaker_recognition",
                side_effect=fake_recognition,
            ),
        ):
            result = detect_unknown_speakers(
                wav, "fake_voices", hf_token="tk", threshold=threshold
            )

        if expected_empty:
            assert result == {}
        else:
            assert "SPEAKER_00" in result
            assert len(result["SPEAKER_00"]) > 0
