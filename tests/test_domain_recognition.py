"""
Slice 2 unit tests: assign_speakers como funcion pura.

Estilo GOOS-sin-mocks: solo numpy + value objects de dominio. Cero I/O.
Verificacion de salida y estado, no de colaboracion.
"""

import numpy as np
import pytest


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float64)
    return v / np.linalg.norm(v)


def _make_transcript(*tags: str):
    """Helper: crea un Transcript con un segmento por tag."""
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )

    segments = tuple(
        TranscriptSegment(
            start_ms=i * 1000,
            end_ms=(i + 1) * 1000,
            text=f"texto-{i}",
            speaker=SpeakerIdentity(diarization_tag=tag),
        )
        for i, tag in enumerate(tags)
    )
    return Transcript(segments=segments, audio_path="x.wav", language="es")


# ── Casos basicos ────────────────────────────────────────────────────────────


class TestAssignSpeakersBasic:
    def test_perfect_match_assigns_name(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        library = {"Manuel": _unit(1.0, 0.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        spk = result.segments[0].speaker
        assert spk.recognized_name == "Manuel"
        assert spk.similarity == pytest.approx(1.0, abs=1e-9)
        assert spk.diarization_tag == "SPEAKER_00"

    def test_picks_best_match_among_multiple(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.1, 0.0)}
        library = {
            "Manuel": _unit(1.0, 0.0, 0.0),   # ~0.995
            "Pamela": _unit(0.0, 1.0, 0.0),   # ~0.0995
            "Agustin": _unit(0.0, 0.0, 1.0),  # 0.0
        }

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        assert result.segments[0].speaker.recognized_name == "Manuel"


# ── Casos limite (anti-bug) ──────────────────────────────────────────────────


class TestAssignSpeakersFallback:
    def test_below_threshold_keeps_diarization_tag(self):
        """Invariante critico: similarity bajo threshold → recognized_name=None,
        label=tag. Jamas el literal 'unknown'."""
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_07")
        embeddings = {"SPEAKER_07": _unit(1.0, 0.0)}
        library = {"Pamela": _unit(0.0, 1.0)}  # ortogonal → similarity = 0

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        spk = result.segments[0].speaker
        assert spk.recognized_name is None
        assert spk.label == "SPEAKER_07"
        assert spk.label != "unknown"

    def test_empty_library_keeps_all_tags(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00", "SPEAKER_01")
        embeddings = {
            "SPEAKER_00": _unit(1.0, 0.0),
            "SPEAKER_01": _unit(0.0, 1.0),
        }
        library: dict[str, np.ndarray] = {}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        for seg in result.segments:
            assert seg.speaker.recognized_name is None
            assert seg.speaker.label == seg.speaker.diarization_tag

    def test_missing_embedding_for_tag_leaves_segment_untouched(self):
        """Si no hay embedding para un tag, el segmento se preserva tal cual.
        Permite re-evaluar parcialmente sin perder identidades previas."""
        from speechlib.domain.recognition import assign_speakers
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )

        transcript = Transcript(
            segments=(
                TranscriptSegment(
                    0, 1000, "previo",
                    SpeakerIdentity(
                        diarization_tag="SPEAKER_00",
                        recognized_name="Manuel",
                        similarity=0.8,
                    ),
                ),
            ),
            audio_path="x.wav",
            language="es",
        )
        embeddings: dict[str, np.ndarray] = {}  # SPEAKER_00 ausente
        library = {"Manuel": _unit(1.0, 0.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        # Sin tocar
        assert result.segments[0].speaker.recognized_name == "Manuel"
        assert result.segments[0].speaker.similarity == 0.8

    def test_re_evaluation_can_clear_previous_identification(self):
        """Si un segmento ya tenia name pero la nueva evaluacion no supera
        threshold, debe limpiarse a None (caso de --all-speakers fix)."""
        from speechlib.domain.recognition import assign_speakers
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )

        transcript = Transcript(
            segments=(
                TranscriptSegment(
                    0, 1000, "x",
                    SpeakerIdentity(
                        diarization_tag="SPEAKER_03",
                        recognized_name="WrongPerson",
                        similarity=0.42,
                    ),
                ),
            ),
            audio_path="x.wav",
            language="es",
        )
        embeddings = {"SPEAKER_03": _unit(1.0, 0.0)}
        library = {"WrongPerson": _unit(0.0, 1.0)}  # ortogonal

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        spk = result.segments[0].speaker
        assert spk.recognized_name is None
        assert spk.label == "SPEAKER_03"  # invariante anti-bug


# ── Coherencia por tag (varios segmentos del mismo speaker) ──────────────────


class TestAssignSpeakersByTag:
    def test_all_segments_with_same_tag_get_same_identity(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript(
            "SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00"
        )
        embeddings = {
            "SPEAKER_00": _unit(1.0, 0.0),
            "SPEAKER_01": _unit(0.0, 1.0),
        }
        library = {"Manuel": _unit(1.0, 0.0), "Pamela": _unit(0.0, 1.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        names = [s.speaker.recognized_name for s in result.segments]
        assert names == ["Manuel", "Pamela", "Manuel", "Manuel"]

    def test_threshold_exact_boundary_is_match(self):
        """similarity == threshold debe contar como match (>=, no >)."""
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        # Vectores con similarity exactamente 0.5
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        # cos(60°) = 0.5
        library = {"X": _unit(0.5, np.sqrt(3) / 2)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.5)

        assert result.segments[0].speaker.recognized_name == "X"


# ── Pureza ───────────────────────────────────────────────────────────────────


class TestAssignSpeakersPurity:
    def test_input_transcript_not_mutated(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        library = {"X": _unit(1.0, 0.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        assert transcript.segments[0].speaker.recognized_name is None
        assert result.segments[0].speaker.recognized_name == "X"
        assert result is not transcript

    def test_metadata_preserved(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        library = {"X": _unit(1.0, 0.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        assert result.audio_path == transcript.audio_path
        assert result.language == transcript.language
        assert result.segments[0].start_ms == transcript.segments[0].start_ms
        assert result.segments[0].end_ms == transcript.segments[0].end_ms
        assert result.segments[0].text == transcript.segments[0].text
