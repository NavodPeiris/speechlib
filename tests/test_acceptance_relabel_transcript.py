"""
Slice 6 AT: relabel_transcript es un servicio que carga un transcript.json
existente, recalcula identidades contra una libreria de voces y guarda el
transcript actualizado. Reemplaza la logica de relabel_vtt --all-speakers.

INVARIANTE ANTI-BUG: ningun segmento queda con label='unknown'. Los que no
matchean conservan su SPEAKER_XX original (el bug que motivo todo el refactor).

Test puro: solo Transcript + numpy + tmp_path. Sin audio, sin pyannote.
"""

import numpy as np
import pytest


def _unit(*c):
    v = np.array(c, dtype=np.float64)
    return v / np.linalg.norm(v)


def _save_transcript_to(path, segments_spec, audio_path="x.wav", language="es"):
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )

    segments = tuple(
        TranscriptSegment(
            start_ms=s["start_ms"],
            end_ms=s["end_ms"],
            text=s.get("text", "x"),
            speaker=SpeakerIdentity(
                diarization_tag=s["tag"],
                recognized_name=s.get("name"),
                similarity=s.get("similarity"),
            ),
        )
        for s in segments_spec
    )
    transcript = Transcript(segments=segments, audio_path=audio_path, language=language)
    transcript.save(path)
    return transcript


def test_relabel_transcript_re_identifies_segment_and_preserves_unmatched(tmp_path):
    """Caso completo del bug original: re-evaluar todos los segmentos contra
    una libreria; los que matchean obtienen recognized_name; los que no
    matchean conservan SPEAKER_XX (jamas 'unknown')."""
    from speechlib.domain.transcript import Transcript
    from speechlib.services.relabel import relabel_transcript

    src = tmp_path / "transcript.json"
    _save_transcript_to(
        src,
        segments_spec=[
            {"start_ms": 0,    "end_ms": 1000, "tag": "SPEAKER_00"},
            {"start_ms": 1000, "end_ms": 2000, "tag": "SPEAKER_01"},
            {"start_ms": 2000, "end_ms": 3000, "tag": "SPEAKER_00"},
        ],
    )

    embeddings_by_tag = {
        "SPEAKER_00": _unit(1.0, 0.0, 0.0),  # Manuel
        "SPEAKER_01": _unit(0.0, 0.0, 1.0),  # ortogonal a todo
    }
    voice_library = {
        "Manuel": _unit(1.0, 0.0, 0.0),
        "Pamela": _unit(0.0, 1.0, 0.0),
    }

    out = tmp_path / "transcript_relabeled.json"
    result = relabel_transcript(
        src=src,
        dst=out,
        embeddings_by_tag=embeddings_by_tag,
        voice_library=voice_library,
        threshold=0.40,
    )

    # SPEAKER_00 ahora es Manuel
    assert result.segments[0].speaker.recognized_name == "Manuel"
    assert result.segments[2].speaker.recognized_name == "Manuel"

    # SPEAKER_01 NO supera threshold → conserva tag, NO 'unknown' (anti-bug)
    s1 = result.segments[1]
    assert s1.speaker.recognized_name is None
    assert s1.speaker.label == "SPEAKER_01"
    assert s1.speaker.label != "unknown"

    # Round-trip al disco preservado
    loaded = Transcript.load(out)
    assert loaded == result


def test_relabel_can_clear_previous_misidentification(tmp_path):
    """Si un segmento ya tenia name (mal asignado) y la nueva evaluacion no
    pasa threshold, debe limpiarse a None y caer al SPEAKER_XX. Este es
    exactamente el caso que el flag --all-speakers debia cubrir."""
    from speechlib.services.relabel import relabel_transcript

    src = tmp_path / "transcript.json"
    _save_transcript_to(
        src,
        segments_spec=[
            {
                "start_ms": 0,
                "end_ms": 1000,
                "tag": "SPEAKER_03",
                "name": "WrongPerson",
                "similarity": 0.42,
            },
        ],
    )
    embeddings_by_tag = {"SPEAKER_03": _unit(1.0, 0.0)}
    voice_library = {"WrongPerson": _unit(0.0, 1.0)}  # ortogonal

    result = relabel_transcript(
        src=src,
        dst=tmp_path / "out.json",
        embeddings_by_tag=embeddings_by_tag,
        voice_library=voice_library,
        threshold=0.40,
    )
    spk = result.segments[0].speaker
    assert spk.recognized_name is None
    assert spk.label == "SPEAKER_03"
    assert spk.label != "unknown"
