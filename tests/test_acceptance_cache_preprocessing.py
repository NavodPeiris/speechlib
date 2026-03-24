"""
AT: si artifacts_dir/16k.wav ya existe, el pipeline no lo regenera (cache hit).
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
def cache_run(tmp_path_factory, request):
    filename, duration_s, framerate = request.param
    tmp = tmp_path_factory.mktemp("cache")
    src = make_tone_wav(tmp / filename, duration_s=duration_s, framerate=framerate)

    from speechlib.core_analysis import core_analysis

    def run():
        core_analysis(
            str(src),
            voices_folder=str(VOICES),
            log_folder=str(tmp),
            language="en",
            modelSize="base",
            ACCESS_TOKEN=os.environ.get("HF_TOKEN"),
            skip_enhance=True,
        )

    run()
    artifacts_dir = src.parent / f".{src.stem}"
    canonical = artifacts_dir / "16k.wav"
    mtime_first = canonical.stat().st_mtime

    run()
    mtime_second = canonical.stat().st_mtime

    return {"canonical": canonical, "mtime_first": mtime_first, "mtime_second": mtime_second}


def test_canonical_16k_exists_after_first_run(cache_run):
    assert cache_run["canonical"].exists()


def test_16k_not_regenerated_on_second_run(cache_run):
    """Segunda ejecucion no modifica artifacts_dir/16k.wav (cache hit)."""
    assert cache_run["mtime_second"] == cache_run["mtime_first"], (
        f"16k.wav fue regenerado: mtime cambio de {cache_run['mtime_first']} "
        f"a {cache_run['mtime_second']}"
    )
