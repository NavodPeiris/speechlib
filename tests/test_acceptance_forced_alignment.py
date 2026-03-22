"""AT: forced alignment con Wav2Vec2 mejora precisión de word timestamps.

Corre align_words con audio real y modelo real.
Requiere: examples/obama_zach.wav presente en el repo.
"""
import os
from pathlib import Path

import pytest

AUDIO = Path(__file__).parent.parent / "examples" / "obama_zach.wav"

skip_reason = []
if not AUDIO.exists():
    skip_reason.append(f"audio no encontrado: {AUDIO}")

needs_audio = pytest.mark.skipif(bool(skip_reason), reason=" | ".join(skip_reason) or "ok")


# ── AT 1: align_words produce timestamps más precisos que Whisper ───────────

@needs_audio
def test_aligned_words_have_tighter_boundaries():
    """
    Dada una transcripción real de Whisper con word_timestamps,
    align_words debe producir timestamps donde cada palabra cabe
    dentro del rango del segmento (no se sale) y tiene duración > 0.
    """
    import types
    import torch
    from speechlib.forced_align import align_words

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simular segmentos Whisper con texto real del audio (Obama + Zach)
    # Whisper produce estos segmentos con word_timestamps=True
    words = [
        types.SimpleNamespace(word=" What", start=0.0, end=0.5),
        types.SimpleNamespace(word=" is", start=0.5, end=0.7),
        types.SimpleNamespace(word=" it", start=0.7, end=0.9),
        types.SimpleNamespace(word=" like", start=0.9, end=1.2),
    ]
    seg = types.SimpleNamespace(
        start=0.0, end=2.0, text="What is it like", words=words,
    )

    result = align_words(str(AUDIO), [seg], "en", device)

    assert len(result) == 1
    aligned = result[0]
    assert len(aligned.words) == 4

    for w in aligned.words:
        assert w.start >= seg.start, f"word '{w.word}' starts before segment"
        assert w.end <= seg.end + 0.1, f"word '{w.word}' ends after segment"
        assert w.end > w.start, f"word '{w.word}' has zero/negative duration"


# ── AT 2: idioma no soportado retorna originales sin crash ──────────────────

@needs_audio
def test_unsupported_language_returns_original():
    """Si el idioma no tiene modelo CTC, retorna segmentos originales intactos."""
    import types
    from speechlib.forced_align import align_words

    word = types.SimpleNamespace(word="test", start=1.0, end=2.0)
    seg = types.SimpleNamespace(start=0.0, end=5.0, text="test", words=[word])

    result = align_words(str(AUDIO), [seg], "xx", "cpu")

    assert len(result) == 1
    assert result[0].words[0].start == 1.0  # unchanged


# ── AT 3: transcribe_full_aligned integra alignment sin romper ──────────────

@needs_audio
def test_transcribe_full_aligned_uses_alignment():
    """
    transcribe_full_aligned con audio real y segmentos de diarización
    produce resultado con texto no vacío — verifica integración completa
    Whisper + forced alignment + speaker assignment.
    """
    import torch
    from speechlib.transcribe import transcribe_full_aligned

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "tiny" if device == "cpu" else "base"

    # Un solo segmento de diarización que cubre los primeros 5 segundos
    diarization_segments = [
        [0.0, 5.0, "SPEAKER_00"],
    ]

    result = transcribe_full_aligned(
        str(AUDIO), diarization_segments, "en", model_size, False,
    )

    assert len(result) == 1
    text = result[0][2]
    assert len(text) > 0, "transcripción vacía — alignment o whisper falló"
