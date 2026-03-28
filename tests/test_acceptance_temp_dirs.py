"""
AT: speaker_recognition usa artifacts_dir/tmp/ en lugar de CWD/temp/.
Llama a la funcion directamente con un segmento sintetico para forzar su ejecucion.
"""

import pytest
from pathlib import Path
from conftest import make_tone_wav
from speechlib.audio_state import AudioState
from speechlib.resample_to_16k import resample_to_16k
from speechlib.speaker_recognition import speaker_recognition

pytestmark = pytest.mark.e2e

VOICES = Path("examples/voices")


@pytest.fixture(
    scope="module",
    params=[("audio.wav", 5.0, 44100)],
    ids=["audio_5s_44100"],
)
def sr_run(tmp_path_factory, request):
    filename, duration_s, framerate = request.param
    tmp = tmp_path_factory.mktemp("tempdirs")
    src = make_tone_wav(tmp / filename, duration_s=duration_s, framerate=framerate)
    state = AudioState(source_path=src, working_path=src)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    state = resample_to_16k(state)

    # Forzar ejecucion de speaker_recognition con un segmento artificial
    segments = [[0.0, 3.0, "SPEAKER_00"]]
    speaker_recognition(str(state.working_path), str(VOICES), segments)

    return {"artifacts_dir": state.artifacts_dir, "working_path": state.working_path}


def test_tmp_dir_is_inside_artifacts_dir(sr_run):
    """speaker_recognition crea artifacts_dir/tmp/ (no CWD/temp/)."""
    assert (sr_run["artifacts_dir"] / "tmp").exists(), (
        "artifacts_dir/tmp/ no fue creado por speaker_recognition"
    )
