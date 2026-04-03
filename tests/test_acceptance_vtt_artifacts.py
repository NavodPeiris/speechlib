"""
AT: transcript VTT se escribe en artifacts_dir/transcript_{lang}.vtt (sin timestamp).
"""
import os
import pytest
from pathlib import Path
from conftest import make_tone_wav

VOICES = Path("examples/voices")
pytestmark = pytest.mark.e2e


@pytest.fixture(
    scope="module",
    params=[("audio.wav", 5.0, 44100)],
    ids=["audio_5s_44100"],
)
def run_result(tmp_path_factory, request):
    filename, duration_s, framerate = request.param
    tmp = tmp_path_factory.mktemp("vtttmp")
    src = make_tone_wav(tmp / filename, duration_s=duration_s, framerate=framerate)

    from speechlib.core_analysis import core_analysis
    core_analysis(
        str(src),
        voices_folder=str(VOICES),
        log_folder=str(tmp),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=os.environ.get("HF_TOKEN"),
        skip_enhance=True,
    )
    artifacts_dir = src.parent / f".{src.stem}"
    return {"artifacts_dir": artifacts_dir, "log_folder": tmp}


def test_transcript_vtt_in_artifacts_dir(run_result):
    """VTT se escribe en artifacts_dir/transcript_en.vtt."""
    assert (run_result["artifacts_dir"] / "transcript_en.vtt").exists(), (
        "artifacts_dir/transcript_en.vtt no fue creado"
    )


def test_no_timestamped_vtt_in_artifacts_dir(run_result):
    """No hay VTT con timestamp en artifacts_dir/."""
    vtts = list(run_result["artifacts_dir"].glob("*_*_*.vtt"))
    assert vtts == [], f"VTT con timestamp encontrado en artifacts_dir: {vtts}"
