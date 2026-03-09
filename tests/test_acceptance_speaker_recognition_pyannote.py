"""
AT: Verify speaker_recognition uses pyannote.embedding instead of speechbrain.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
from conftest import make_wav


def test_speaker_recognition_uses_pyannote_embedding(tmp_path):
    """
    speaker_recognition debe usar pyannote.audio.Model (embedding)
    en lugar de speechbrain.SpeakerRecognition.
    """
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()

    obama_dir = voices_dir / "obama"
    obama_dir.mkdir()
    obama_ref = make_wav(obama_dir / "obama_ref.wav", n_frames=16000)

    audio_file = make_wav(tmp_path / "audio.wav", n_frames=16000)

    with (
        patch("speechlib.speaker_recognition._get_inference") as mock_inference_fn,
        patch("speechlib.speaker_recognition.slice_and_save"),
        patch("speechlib.speaker_recognition.os.remove"),
    ):
        mock_inference_fn.return_value = lambda path: np.array([[1.0, 0.0, 0.0]])

        from speechlib.speaker_recognition import speaker_recognition

        segments = [[0.0, 1.0, "SPEAKER_00"], [1.0, 2.0, "SPEAKER_00"]]
        result = speaker_recognition(
            str(audio_file), str(voices_dir), segments, wildcards=[]
        )

        mock_inference_fn.assert_called()
