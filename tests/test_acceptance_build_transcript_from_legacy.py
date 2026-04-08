"""
Slice 5 AT: build_transcript_from_legacy_segments convierte el formato
legacy de core_analysis (lista de [start_s, end_s, text, label]) en un
Transcript del nuevo dominio, preservando el diarization_tag de pyannote
para los speakers no identificados (invariante anti-bug).

Test puro: solo listas/tuples sinteticas + value objects. Sin core_analysis,
sin pyannote, sin audio.
"""

import pytest


def test_build_transcript_preserves_diarization_tag_for_unidentified():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    # Formato legacy de core_analysis: [start_s, end_s, text, label]
    legacy_segments = [
        [0.0, 2.5, "hola", "Manuel Olguin"],
        [2.5, 5.0, "extraño", "SPEAKER_03"],
        [5.0, 7.5, "como estas", "Manuel Olguin"],
    ]
    # Annotation simplificada como lista de turnos pyannote (start_s, end_s, tag)
    annotation_turns = [
        (0.0, 2.5, "SPEAKER_00"),
        (2.5, 5.0, "SPEAKER_03"),
        (5.0, 7.5, "SPEAKER_00"),
    ]
    speaker_map = {
        "SPEAKER_00": "Manuel Olguin",
        "SPEAKER_03": "SPEAKER_03",  # no identificado
    }

    transcript = build_transcript_from_legacy_segments(
        legacy_segments=legacy_segments,
        annotation_turns=annotation_turns,
        speaker_map=speaker_map,
        audio_path="rec.wav",
        language="es",
    )

    assert len(transcript.segments) == 3

    # Segmento 0: Manuel identificado, tag preservado
    s0 = transcript.segments[0]
    assert s0.start_ms == 0
    assert s0.end_ms == 2500
    assert s0.text == "hola"
    assert s0.speaker.diarization_tag == "SPEAKER_00"
    assert s0.speaker.recognized_name == "Manuel Olguin"
    assert s0.speaker.label == "Manuel Olguin"

    # Segmento 1: SPEAKER_03 no identificado, label es el tag
    s1 = transcript.segments[1]
    assert s1.speaker.diarization_tag == "SPEAKER_03"
    assert s1.speaker.recognized_name is None
    assert s1.speaker.label == "SPEAKER_03"
    assert s1.speaker.label != "unknown"  # invariante

    # Segmento 2: Manuel otra vez
    s2 = transcript.segments[2]
    assert s2.speaker.diarization_tag == "SPEAKER_00"
    assert s2.speaker.recognized_name == "Manuel Olguin"


def test_build_transcript_metadata():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    transcript = build_transcript_from_legacy_segments(
        legacy_segments=[],
        annotation_turns=[],
        speaker_map={},
        audio_path="audio_xyz.wav",
        language="es",
    )

    assert transcript.audio_path == "audio_xyz.wav"
    assert transcript.language == "es"
    assert transcript.segments == ()
