"""
Slice 3 AT: Transcript.save / Transcript.load round-trip.

El aggregate Transcript es la fuente canonica del refactor. Debe poder
serializarse y deserializarse preservando TODOS los campos (incluyendo
similarity, que el VTT pierde). Esto habilita que el VTT pase a ser un
render derivado en lugar de la fuente de verdad.

Test puro: solo tmp_path + value objects. Sin pyannote ni audio.
"""

import pytest


def test_save_and_load_preserves_all_fields(tmp_path):
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )

    original = Transcript(
        segments=(
            TranscriptSegment(
                start_ms=0,
                end_ms=1500,
                text="hola que tal",
                speaker=SpeakerIdentity(diarization_tag="SPEAKER_00"),
            ),
            TranscriptSegment(
                start_ms=1500,
                end_ms=3200,
                text="bien y tu",
                speaker=SpeakerIdentity(
                    diarization_tag="SPEAKER_01",
                    recognized_name="Manuel Olguin",
                    similarity=0.7234,
                ),
            ),
        ),
        audio_path="recording.wav",
        language="es",
    )

    path = tmp_path / "transcript.json"
    original.save(path)
    loaded = Transcript.load(path)

    assert loaded == original


def test_load_after_save_preserves_invariant(tmp_path):
    """Invariante anti-bug sobrevive al round-trip: segmentos no
    identificados conservan su SPEAKER_XX, jamas 'unknown'."""
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )

    original = Transcript(
        segments=(
            TranscriptSegment(
                0, 1000, "x",
                SpeakerIdentity(diarization_tag="SPEAKER_07"),
            ),
        ),
        audio_path="r.wav",
        language="es",
    )

    path = tmp_path / "t.json"
    original.save(path)
    loaded = Transcript.load(path)

    assert loaded.segments[0].speaker.label == "SPEAKER_07"
    assert loaded.segments[0].speaker.label != "unknown"
    assert loaded.segments[0].speaker.recognized_name is None
