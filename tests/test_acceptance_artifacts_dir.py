"""
AT: core_analysis crea la carpeta artifacts_dir junto al audio source.
"""
import os
import pytest
from pathlib import Path
from conftest import make_tone_wav


VOICES = Path("examples/voices")
pytestmark = pytest.mark.e2e


@pytest.fixture(
    scope="module",
    params=[("audio.wav", 5.0)],
    ids=["audio_5s"],
)
def artifacts_dir(tmp_path_factory, request):
    """Corre core_analysis sobre audio sintetico y retorna artifacts_dir."""
    filename, duration_s = request.param
    tmp = tmp_path_factory.mktemp("artifacts")
    src = make_tone_wav(tmp / filename, duration_s=duration_s)

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
    return src.parent / f".{src.stem}"


def test_artifacts_dir_is_created(artifacts_dir):
    """core_analysis crea .{stem}/ junto al source."""
    assert artifacts_dir.exists(), f"artifacts_dir no existe: {artifacts_dir}"
    assert artifacts_dir.is_dir()


def test_artifacts_dir_is_hidden(artifacts_dir):
    """La carpeta empieza con punto (convencion oculta)."""
    assert artifacts_dir.name.startswith(".")
