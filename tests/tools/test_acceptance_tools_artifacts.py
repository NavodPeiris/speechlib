"""
AT: tools usan artifacts_dir/ para sus outputs (Slice 8).
- batch_process: unknown speakers → artifacts_dir/unknown_speakers/
"""
from pathlib import Path
from unittest.mock import patch
import pytest
from conftest import make_tone_wav


def _make_voices_dir(base: Path) -> Path:
    voices = base / "voices"
    (voices / "Agustin").mkdir(parents=True)
    make_tone_wav(voices / "Agustin" / "segment_01.wav", duration_s=2.0)
    return voices


def test_batch_process_unknown_clips_go_to_artifacts_dir(tmp_path):
    """batch_process extrae unknown speakers a artifacts_dir/unknown_speakers/."""
    from speechlib.tools.batch_process import batch_process

    folder = tmp_path / "session"
    folder.mkdir()
    audio = make_tone_wav(folder / "meeting.wav", duration_s=10.0)
    voices = _make_voices_dir(tmp_path)

    # core_analysis mocked (requires GPU/HF): returns one unknown segment
    with patch("speechlib.tools.batch_process.core_analysis") as mock_ca:
        mock_ca.return_value = [[0.0, 5.0, "hello", "unknown"]]
        report = batch_process(
            folders=[folder],
            voices_folder=voices,
            language="es",
            access_token="fake_token",
        )

    artifacts_dir = folder / f".{audio.stem}"
    assert (artifacts_dir / "unknown_speakers").exists(), (
        f"artifacts_dir/unknown_speakers/ not created. "
        f"artifacts_dir contents: {list(artifacts_dir.iterdir()) if artifacts_dir.exists() else 'dir missing'}"
    )
