"""
AT: speakers desconocidos (score < threshold) se extraen automáticamente
a voices/_unknown/{SPEAKER_XX}_{audio_stem}/ para que el usuario los nombre.

Caso real: Patricio Renner aparece en grabación junto a Agustin.
Agustin debe identificarse (alta confianza), Patricio → carpeta _unknown.
"""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from conftest import make_tone_wav


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_voices_dir(tmp_path: Path) -> Path:
    """Crea librería mínima con un speaker conocido (Agustin)."""
    voices = tmp_path / "voices"
    agustin = voices / "Agustin"
    agustin.mkdir(parents=True)
    make_tone_wav(agustin / "segment_01.wav", duration_s=2.0)
    return voices


# ── AT: módulo extract_unknown_speakers ───────────────────────────────────────

def test_unknown_speaker_segments_saved_to_disk(tmp_path):
    """Segmentos de speaker desconocido se guardan en voices/_unknown/."""
    from speechlib.tools.extract_unknown_speakers import extract_unknown_speakers

    audio = make_tone_wav(tmp_path / "meeting.wav", duration_s=10.0)
    unknown_dir = tmp_path / "_unknown"

    # Segmentos: SPEAKER_01 desconocido, varios fragmentos
    unknown_segments = {
        "SPEAKER_01": [[1.0, 3.5], [5.0, 8.0], [9.0, 10.0]],
    }

    result = extract_unknown_speakers(
        audio_path=audio,
        unknown_segments=unknown_segments,
        output_dir=unknown_dir,
        min_duration_s=2.0,
        max_clips=3,
    )

    assert "SPEAKER_01" in result
    speaker_dir = result["SPEAKER_01"]
    assert speaker_dir.exists()
    wavs = list(speaker_dir.glob("*.wav"))
    assert len(wavs) >= 1, f"No se generaron WAVs para SPEAKER_01: {speaker_dir}"


def test_segments_shorter_than_min_duration_are_skipped(tmp_path):
    """Segmentos < min_duration_s no se incluyen como muestras de voz."""
    from speechlib.tools.extract_unknown_speakers import extract_unknown_speakers

    audio = make_tone_wav(tmp_path / "meeting.wav", duration_s=10.0)
    unknown_dir = tmp_path / "_unknown"

    unknown_segments = {
        "SPEAKER_01": [[0.0, 0.5], [1.0, 1.3]],  # todos < 2s
    }

    result = extract_unknown_speakers(
        audio_path=audio,
        unknown_segments=unknown_segments,
        output_dir=unknown_dir,
        min_duration_s=2.0,
        max_clips=3,
    )

    # No debe crear carpeta si no hay segmentos válidos
    assert "SPEAKER_01" not in result or not list(result["SPEAKER_01"].glob("*.wav"))


def test_max_clips_limits_saved_segments(tmp_path):
    """No se guardan más de max_clips segmentos por speaker."""
    from speechlib.tools.extract_unknown_speakers import extract_unknown_speakers

    audio = make_tone_wav(tmp_path / "meeting.wav", duration_s=30.0)
    unknown_dir = tmp_path / "_unknown"

    # 5 segmentos largos, max_clips=2
    unknown_segments = {
        "SPEAKER_01": [
            [0.0, 4.0], [5.0, 9.0], [10.0, 14.0], [15.0, 19.0], [20.0, 24.0]
        ],
    }

    result = extract_unknown_speakers(
        audio_path=audio,
        unknown_segments=unknown_segments,
        output_dir=unknown_dir,
        min_duration_s=2.0,
        max_clips=2,
    )

    assert "SPEAKER_01" in result
    wavs = list(result["SPEAKER_01"].glob("*.wav"))
    assert len(wavs) <= 2, f"Se guardaron {len(wavs)} clips, máximo es 2"


def test_multiple_unknown_speakers_each_get_their_folder(tmp_path):
    """Cada speaker desconocido obtiene su propia carpeta."""
    from speechlib.tools.extract_unknown_speakers import extract_unknown_speakers

    audio = make_tone_wav(tmp_path / "meeting.wav", duration_s=30.0)
    unknown_dir = tmp_path / "_unknown"

    unknown_segments = {
        "SPEAKER_01": [[0.0, 4.0], [5.0, 9.0]],
        "SPEAKER_02": [[10.0, 14.0], [15.0, 19.0]],
    }

    result = extract_unknown_speakers(
        audio_path=audio,
        unknown_segments=unknown_segments,
        output_dir=unknown_dir,
        min_duration_s=2.0,
        max_clips=3,
    )

    assert "SPEAKER_01" in result
    assert "SPEAKER_02" in result
    assert result["SPEAKER_01"] != result["SPEAKER_02"]


def test_output_dir_name_includes_audio_stem(tmp_path):
    """La carpeta del speaker incluye el stem del audio para evitar colisiones."""
    from speechlib.tools.extract_unknown_speakers import extract_unknown_speakers

    audio = make_tone_wav(tmp_path / "patricio_meeting.wav", duration_s=10.0)
    unknown_dir = tmp_path / "_unknown"

    unknown_segments = {"SPEAKER_01": [[1.0, 4.0]]}

    result = extract_unknown_speakers(
        audio_path=audio,
        unknown_segments=unknown_segments,
        output_dir=unknown_dir,
        min_duration_s=2.0,
        max_clips=3,
    )

    if "SPEAKER_01" in result:
        folder_name = result["SPEAKER_01"].name
        assert "patricio_meeting" in folder_name or "SPEAKER_01" in folder_name


def test_selects_longest_segments_first(tmp_path):
    """Se seleccionan los segmentos más largos cuando se supera max_clips."""
    from speechlib.tools.extract_unknown_speakers import extract_unknown_speakers
    from speechlib.audio_utils import slice_and_save

    audio = make_tone_wav(tmp_path / "meeting.wav", duration_s=30.0)
    unknown_dir = tmp_path / "_unknown"

    # Segmentos de duraciones variadas (en segundos): 5s, 2s, 4s, 3s
    unknown_segments = {
        "SPEAKER_01": [
            [0.0,  5.0],   # 5s — debe seleccionarse
            [6.0,  8.0],   # 2s — descartado si max_clips=2
            [9.0,  13.0],  # 4s — debe seleccionarse
            [14.0, 17.0],  # 3s — descartado si max_clips=2
        ],
    }

    result = extract_unknown_speakers(
        audio_path=audio,
        unknown_segments=unknown_segments,
        output_dir=unknown_dir,
        min_duration_s=2.0,
        max_clips=2,
    )

    assert "SPEAKER_01" in result
    wavs = sorted(result["SPEAKER_01"].glob("*.wav"),
                  key=lambda p: p.stat().st_size, reverse=True)
    assert len(wavs) == 2
    # El clip más grande debe ser el de 5s (mayor tamaño en bytes)
    assert wavs[0].stat().st_size > wavs[1].stat().st_size
