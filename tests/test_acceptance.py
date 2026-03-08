"""
Acceptance Test: el pipeline de preprocesamiento preserva el source.

Este test permanece en RED hasta que Slice 5 (core_analysis encadenado) esté completo.
"""
import wave
import struct
import pytest
from pathlib import Path
from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav
from speechlib.convert_to_mono import convert_to_mono
from speechlib.re_encode import re_encode


def create_test_wav(tmp_path, channels=2, sampwidth=1, framerate=16000, n_frames=1600):
    """Crea un WAV sintético con los parámetros dados."""
    path = tmp_path / "source.wav"
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        # Datos de silencio
        if sampwidth == 1:
            data = bytes([128] * n_frames * channels)
        else:
            data = struct.pack(f"<{n_frames * channels}h", *([0] * n_frames * channels))
        f.writeframes(data)
    return path


def test_preprocessing_preserves_source(tmp_path):
    """
    Dado un WAV estéreo 8-bit,
    cuando el pipeline completo se ejecuta,
    entonces el source no fue modificado y working_path apunta a un archivo distinto
    con todos los flags en True.
    """
    source = create_test_wav(tmp_path, channels=2, sampwidth=1)
    original_content = source.read_bytes()

    state = AudioState(source_path=source, working_path=source)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)

    assert source.read_bytes() == original_content, "El source fue modificado"
    assert state.working_path != state.source_path, "working_path debe ser distinto al source"
    assert state.working_path.exists(), "working_path debe existir"
    assert state.is_wav is True
    assert state.is_mono is True
    assert state.is_16bit is True
