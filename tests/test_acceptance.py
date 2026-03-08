"""
Acceptance Test: el pipeline de preprocesamiento preserva el source.

Este test permanece en RED hasta que Slice 5 (core_analysis encadenado) esté completo.
"""

from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav
from speechlib.convert_to_mono import convert_to_mono
from speechlib.re_encode import re_encode
from conftest import make_wav


def test_preprocessing_preserves_source(tmp_path):
    """
    Dado un WAV estéreo 8-bit,
    cuando el pipeline completo se ejecuta,
    entonces el source no fue modificado y working_path apunta a un archivo distinto
    con todos los flags en True.
    """
    source = make_wav(tmp_path / "source.wav", channels=2, sampwidth=1, n_frames=1600)
    original_content = source.read_bytes()

    state = AudioState(source_path=source, working_path=source)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)

    assert source.read_bytes() == original_content, "El source fue modificado"
    assert state.working_path != state.source_path, (
        "working_path debe ser distinto al source"
    )
    assert state.working_path.exists(), "working_path debe existir"
    assert state.is_wav is True
    assert state.is_mono is True
    assert state.is_16bit is True
