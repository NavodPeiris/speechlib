"""
AT: artefactos de preprocessing van a artifacts_dir/, no junto al source.
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
def run(tmp_path_factory, request):
    filename, duration_s, framerate = request.param
    tmp = tmp_path_factory.mktemp("preproc")
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
    return {"src": src, "artifacts_dir": src.parent / f".{src.stem}"}


def test_16k_wav_is_inside_artifacts_dir(run):
    """El WAV resampleado a 16kHz está en artifacts_dir/16k.wav."""
    assert (run["artifacts_dir"] / "16k.wav").exists()


def test_source_dir_has_no_intermediate_wav_suffixes(run):
    """La carpeta del source no tiene _mono.wav, _16k.wav, _16bit.wav."""
    src_dir = run["src"].parent
    polluting = [
        f for f in src_dir.iterdir()
        if f.is_file() and any(s in f.name for s in ("_mono", "_16k", "_16bit"))
    ]
    assert polluting == [], f"Archivos intermedios junto al source: {polluting}"
