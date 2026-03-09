"""
Unit tests for speaker_recognition module with pyannote.
"""

from unittest.mock import patch, MagicMock, MagicMock
from pathlib import Path
import numpy as np
from conftest import make_wav


def test_extract_embedding_from_audio(tmp_path):
    """Verify embedding extraction from audio file using pyannote."""
    audio_file = make_wav(tmp_path / "test.wav", n_frames=16000)

    with (
        patch("speechlib.speaker_recognition.Model") as mock_model_cls,
        patch("speechlib.speaker_recognition.Inference") as mock_inference_cls,
    ):
        expected_embedding = np.random.rand(1, 512).astype(np.float32)

        mock_inference = MagicMock()
        mock_inference.return_value = expected_embedding
        mock_inference_cls.return_value = mock_inference

        from speechlib.speaker_recognition import get_embedding

        result = get_embedding(str(audio_file))

        mock_inference_cls.assert_called_once()
        mock_inference.assert_called_once_with(str(audio_file))
        np.testing.assert_array_equal(result, expected_embedding)


def test_cosine_similarity_calculation():
    """Verify cosine similarity between two embeddings."""
    from speechlib.speaker_recognition import cosine_similarity

    embedding1 = np.array([[1.0, 0.0, 0.0]])
    embedding2 = np.array([[1.0, 0.0, 0.0]])

    sim = cosine_similarity(embedding1, embedding2)
    assert sim == 1.0

    embedding3 = np.array([[0.0, 1.0, 0.0]])
    sim2 = cosine_similarity(embedding1, embedding3)
    assert sim2 == 0.0


def test_find_best_speaker():
    """Verify speaker identification based on embedding similarity."""
    from speechlib.speaker_recognition import find_best_speaker

    test_embedding = np.array([[1.0, 0.0, 0.0]])

    speaker_embeddings = {
        "obama": np.array([[1.0, 0.0, 0.0]]),
        "zach": np.array([[0.0, 1.0, 0.0]]),
    }

    best_speaker = find_best_speaker(test_embedding, speaker_embeddings)
    assert best_speaker == "obama"


def test_speaker_recognition_compares_all_voice_files(tmp_path):
    """Verify speaker recognition compares against all voice files in folder."""
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()

    obama_dir = voices_dir / "obama"
    obama_dir.mkdir()
    make_wav(obama_dir / "obama1.wav", n_frames=16000)
    make_wav(obama_dir / "obama2.wav", n_frames=16000)

    zach_dir = voices_dir / "zach"
    zach_dir.mkdir()
    make_wav(zach_dir / "zach1.wav", n_frames=16000)

    audio_file = make_wav(tmp_path / "audio.wav", n_frames=16000)

    with (
        patch("speechlib.speaker_recognition._get_inference") as mock_inference,
        patch("speechlib.speaker_recognition.slice_and_save"),
        patch("speechlib.speaker_recognition.os.remove"),
    ):
        mock_inference.return_value = lambda path: np.array([[1.0, 0.0, 0.0]])

        from speechlib.speaker_recognition import speaker_recognition

        segments = [[0.0, 1.0, "SPEAKER_00"]]
        result = speaker_recognition(
            str(audio_file), str(voices_dir), segments, wildcards=[]
        )

        assert mock_inference.call_count >= 1
