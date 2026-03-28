"""
AT: speaker_recognition muestrea al menos 60 segundos antes de decidir.

Bug original: limit=60 en ms (no segundos). Con un segmento de 5s (5000ms),
duration >= 60 era True en la primera iteracion, causando false positives
por muestreo insuficiente.

Fix: limit=60_000 (60 segundos expresado en ms).
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from conftest import make_tone_wav


def _make_voices(tmp_path: Path) -> Path:
    voices = tmp_path / "voices"
    (voices / "Agustin").mkdir(parents=True)
    make_tone_wav(voices / "Agustin" / "sample.wav", duration_s=2.0)
    return voices


def _fake_slice(call_counter):
    """Mock de slice_and_save: crea archivo dummy sin procesar audio."""

    def _slice(src, start, end, dest):
        call_counter[0] += 1
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"fake_audio")

    return _slice


def _high_score_inference():
    """Mock de _get_inference: siempre devuelve embedding idéntico (score=1.0)."""
    emb = np.array([[1.0, 0.0, 0.0]])
    mock_inf = MagicMock()
    mock_inf.return_value = emb
    return mock_inf


def test_limit_not_exceeded_after_single_short_segment(tmp_path):
    """Con segmentos de 5s y limit=60s, el loop NO rompe en la primera iteracion.

    Antes del fix (limit=60ms): 5000ms >= 60 → break tras segmento 1.
    Despues del fix (limit=60_000ms): 5000ms < 60000 → continua muestreando.
    """
    from speechlib.speaker_recognition import speaker_recognition

    wav = make_tone_wav(tmp_path / "audio.wav", duration_s=30.0)
    voices = _make_voices(tmp_path)

    # 6 segmentos de 5s = 30s total < 60s → con limit=60s NO debe romper
    segments = [[i * 5.0, (i + 1) * 5.0, "SPEAKER_00"] for i in range(6)]
    call_count = [0]

    with patch(
        "speechlib.speaker_recognition.slice_and_save",
        side_effect=_fake_slice(call_count),
    ):
        with patch(
            "speechlib.speaker_recognition._get_inference",
            return_value=_high_score_inference(),
        ):
            speaker_recognition(str(wav), str(voices), segments)

    assert call_count[0] > 1, (
        f"speaker_recognition proceso solo {call_count[0]} segmento(s). "
        f"Con limit=60s y 6 segmentos de 5s (30s total), debe procesar todos."
    )


def test_limit_breaks_after_60_seconds_of_audio(tmp_path):
    """Con >60s de audio identificado con confianza, el loop debe romper (eficiencia)."""
    from speechlib.speaker_recognition import speaker_recognition

    wav = make_tone_wav(tmp_path / "audio.wav", duration_s=120.0)
    voices = _make_voices(tmp_path)

    # 20 segmentos de 5s = 100s > 60s → debe romper antes de procesar todos
    segments = [[i * 5.0, (i + 1) * 5.0, "SPEAKER_00"] for i in range(20)]
    call_count = [0]

    with patch(
        "speechlib.speaker_recognition.slice_and_save",
        side_effect=_fake_slice(call_count),
    ):
        with patch(
            "speechlib.speaker_recognition._get_inference",
            return_value=_high_score_inference(),
        ):
            speaker_recognition(str(wav), str(voices), segments)

    # Con limit=60_000ms y segmentos de 5s, rompe ~segmento 12 (12*5000=60000ms)
    assert call_count[0] < 20, (
        f"speaker_recognition proceso los {call_count[0]} segmentos sin romper. "
        f"Con limit=60s y 100s de audio, debe romper antes."
    )
    assert call_count[0] >= 12, (
        f"speaker_recognition proceso solo {call_count[0]} segmentos. "
        f"Con segmentos de 5s necesita ~12 para alcanzar 60s."
    )
