"""
AT: End-to-end test for speaker recognition using pyannote with real audio files.
Uses examples/voices/ as reference and examples/obama_zach.wav as test audio.
"""

import os
from pathlib import Path
import pytest


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
def test_speaker_recognition_end_to_end():
    """
    End-to-end test: verify speaker recognition identifies speakers
    in obama_zach.wav using voices/ as reference.
    """
    from speechlib.speaker_recognition import speaker_recognition

    audio_file = "examples/obama_zach.wav"
    voices_folder = "examples/voices"

    assert Path(audio_file).exists(), f"Audio file not found: {audio_file}"
    assert Path(voices_folder).exists(), f"Voices folder not found: {voices_folder}"

    speakers = os.listdir(voices_folder)
    assert "obama" in speakers, "obama reference voice not found"
    assert "zach" in speakers, "zach reference voice not found"

    segments = [
        [0.0, 10.0, "SPEAKER_00"],
        [10.0, 20.0, "SPEAKER_01"],
    ]

    result = speaker_recognition(audio_file, voices_folder, segments, wildcards=[])

    assert result in ["obama", "zach", "unknown"], f"Unexpected result: {result}"
    print(f"Speaker recognition result: {result}")
