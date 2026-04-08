"""
Slice 2 AT: assign_speakers es una funcion pura que asigna identidades
contra una libreria de voces, preservando el diarization_tag cuando no
hay match.

Test puro: solo numpy arrays sinteticos + value objects de dominio.
Sin pyannote, sin audio, sin filesystem, sin mocks.

Invariante anti-bug ejercitado de extremo a extremo en el aggregate:
los segmentos sin match conservan su SPEAKER_XX en label, jamas "unknown".
"""

import numpy as np


def _unit(*components: float) -> np.ndarray:
    """Vector unitario para que cosine similarity sea exactamente cos(angulo)."""
    v = np.array(components, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_assign_speakers_identifies_known_voice_and_preserves_tag_for_unknown():
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )
    from speechlib.domain.recognition import assign_speakers

    # 3 segmentos: 2 del SPEAKER_00 (Manuel), 1 del SPEAKER_01 (desconocido)
    transcript = Transcript(
        segments=(
            TranscriptSegment(
                0, 1000, "hola",
                SpeakerIdentity(diarization_tag="SPEAKER_00"),
            ),
            TranscriptSegment(
                1000, 2000, "extranio",
                SpeakerIdentity(diarization_tag="SPEAKER_01"),
            ),
            TranscriptSegment(
                2000, 3000, "como estas",
                SpeakerIdentity(diarization_tag="SPEAKER_00"),
            ),
        ),
        audio_path="recording.wav",
        language="es",
    )

    # SPEAKER_00 embedding ~ Manuel; SPEAKER_01 ortogonal a todo
    embeddings_by_tag = {
        "SPEAKER_00": _unit(1.0, 0.0, 0.0),
        "SPEAKER_01": _unit(0.0, 0.0, 1.0),
    }
    voice_library = {
        "Manuel": _unit(1.0, 0.0, 0.0),
        "Pamela": _unit(0.0, 1.0, 0.0),
    }

    result = assign_speakers(
        transcript, embeddings_by_tag, voice_library, threshold=0.40
    )

    # SPEAKER_00 → identificado como Manuel en ambos segmentos
    assert result.segments[0].speaker.recognized_name == "Manuel"
    assert result.segments[2].speaker.recognized_name == "Manuel"
    assert result.segments[0].speaker.label == "Manuel"

    # SPEAKER_01 → no superó threshold → conserva tag pyannote, NO "unknown"
    assert result.segments[1].speaker.recognized_name is None
    assert result.segments[1].speaker.label == "SPEAKER_01"
    assert result.segments[1].speaker.label != "unknown"


def test_assign_speakers_returns_new_transcript_without_mutating_input():
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )
    from speechlib.domain.recognition import assign_speakers

    original = Transcript(
        segments=(
            TranscriptSegment(
                0, 1000, "x",
                SpeakerIdentity(diarization_tag="SPEAKER_00"),
            ),
        ),
        audio_path="r.wav",
        language="es",
    )
    embeddings_by_tag = {"SPEAKER_00": _unit(1.0, 0.0)}
    voice_library = {"X": _unit(1.0, 0.0)}

    result = assign_speakers(original, embeddings_by_tag, voice_library, threshold=0.40)

    # Original sin tocar (inmutable)
    assert original.segments[0].speaker.recognized_name is None
    # Resultado nuevo
    assert result is not original
    assert result.segments[0].speaker.recognized_name == "X"
