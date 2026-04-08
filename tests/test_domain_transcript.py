"""
Slice 1 unit tests: domain model (SpeakerIdentity, TranscriptSegment, Transcript).

Tests funcionales puros estilo GOOS-sin-mocks: cero I/O, cero mocks, cero
filesystem. Solo value objects del dominio. Verifican comportamiento y estado,
no colaboracion.
"""

import pytest


# ── SpeakerIdentity ──────────────────────────────────────────────────────────


class TestSpeakerIdentity:
    def test_unidentified_when_no_recognized_name(self):
        from speechlib.domain.transcript import SpeakerIdentity

        speaker = SpeakerIdentity(diarization_tag="SPEAKER_03")

        assert speaker.is_identified is False
        assert speaker.recognized_name is None
        assert speaker.similarity is None

    def test_identified_when_recognized_name_present(self):
        from speechlib.domain.transcript import SpeakerIdentity

        speaker = SpeakerIdentity(
            diarization_tag="SPEAKER_03",
            recognized_name="Pamela Falconi",
            similarity=0.55,
        )

        assert speaker.is_identified is True
        assert speaker.label == "Pamela Falconi"

    def test_label_falls_back_to_tag_when_unidentified(self):
        from speechlib.domain.transcript import SpeakerIdentity

        speaker = SpeakerIdentity(diarization_tag="SPEAKER_11")

        assert speaker.label == "SPEAKER_11"

    def test_with_recognition_returns_new_instance(self):
        from speechlib.domain.transcript import SpeakerIdentity

        original = SpeakerIdentity(diarization_tag="SPEAKER_02")
        updated = original.with_recognition(name="Agustin Villena", similarity=0.61)

        # Inmutable: original no cambia
        assert original.recognized_name is None
        assert original.is_identified is False
        # Nuevo objeto con la actualizacion
        assert updated.recognized_name == "Agustin Villena"
        assert updated.similarity == 0.61
        assert updated.diarization_tag == "SPEAKER_02"
        assert updated.is_identified is True

    def test_with_recognition_can_clear_to_none(self):
        """Re-evaluacion fallida vuelve a unidentified pero conserva el tag."""
        from speechlib.domain.transcript import SpeakerIdentity

        identified = SpeakerIdentity(
            diarization_tag="SPEAKER_04",
            recognized_name="X",
            similarity=0.9,
        )
        cleared = identified.with_recognition(name=None, similarity=0.12)

        assert cleared.recognized_name is None
        assert cleared.is_identified is False
        assert cleared.label == "SPEAKER_04"  # invariante anti-bug

    def test_value_equality(self):
        from speechlib.domain.transcript import SpeakerIdentity

        a = SpeakerIdentity(diarization_tag="SPEAKER_00", recognized_name="X", similarity=0.5)
        b = SpeakerIdentity(diarization_tag="SPEAKER_00", recognized_name="X", similarity=0.5)

        assert a == b
        assert hash(a) == hash(b)

    def test_immutability(self):
        from speechlib.domain.transcript import SpeakerIdentity

        speaker = SpeakerIdentity(diarization_tag="SPEAKER_00")
        with pytest.raises(Exception):  # FrozenInstanceError
            speaker.diarization_tag = "SPEAKER_99"  # type: ignore[misc]


# ── TranscriptSegment ────────────────────────────────────────────────────────


class TestTranscriptSegment:
    def _segment(self, **overrides):
        from speechlib.domain.transcript import SpeakerIdentity, TranscriptSegment

        defaults = dict(
            start_ms=1000,
            end_ms=3500,
            text="hola mundo",
            speaker=SpeakerIdentity(diarization_tag="SPEAKER_00"),
        )
        defaults.update(overrides)
        return TranscriptSegment(**defaults)

    def test_duration_ms(self):
        seg = self._segment(start_ms=1000, end_ms=3500)
        assert seg.duration_ms == 2500

    def test_with_speaker_returns_new_instance(self):
        from speechlib.domain.transcript import SpeakerIdentity

        seg = self._segment()
        new_speaker = SpeakerIdentity(
            diarization_tag="SPEAKER_00",
            recognized_name="Manuel",
            similarity=0.7,
        )
        updated = seg.with_speaker(new_speaker)

        # Inmutable
        assert seg.speaker.recognized_name is None
        # Nuevo objeto con speaker actualizado
        assert updated.speaker.recognized_name == "Manuel"
        assert updated.start_ms == seg.start_ms
        assert updated.text == seg.text

    def test_immutability(self):
        seg = self._segment()
        with pytest.raises(Exception):
            seg.text = "otro"  # type: ignore[misc]

    def test_value_equality(self):
        a = self._segment()
        b = self._segment()
        assert a == b


# ── Transcript ───────────────────────────────────────────────────────────────


class TestTranscript:
    def _transcript(self, segments=None):
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )

        if segments is None:
            segments = (
                TranscriptSegment(
                    start_ms=0,
                    end_ms=1000,
                    text="uno",
                    speaker=SpeakerIdentity(diarization_tag="SPEAKER_00"),
                ),
                TranscriptSegment(
                    start_ms=1000,
                    end_ms=2000,
                    text="dos",
                    speaker=SpeakerIdentity(
                        diarization_tag="SPEAKER_01",
                        recognized_name="Manuel",
                        similarity=0.8,
                    ),
                ),
                TranscriptSegment(
                    start_ms=2000,
                    end_ms=3000,
                    text="tres",
                    speaker=SpeakerIdentity(diarization_tag="SPEAKER_00"),
                ),
            )
        return Transcript(
            segments=segments,
            audio_path="recording.wav",
            language="es",
        )

    def test_diarization_tags_returns_unique_set(self):
        t = self._transcript()
        assert t.diarization_tags == frozenset({"SPEAKER_00", "SPEAKER_01"})

    def test_with_segments_returns_new_instance(self):
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            TranscriptSegment,
        )

        t = self._transcript()
        new_segments = tuple(
            seg.with_speaker(
                SpeakerIdentity(
                    diarization_tag=seg.speaker.diarization_tag,
                    recognized_name="Bulk",
                    similarity=0.99,
                )
            )
            for seg in t.segments
        )
        updated = t.with_segments(new_segments)

        # Original sin tocar
        assert t.segments[0].speaker.recognized_name is None
        # Nuevo con todos identificados
        assert all(s.speaker.recognized_name == "Bulk" for s in updated.segments)
        assert updated.audio_path == t.audio_path
        assert updated.language == t.language

    def test_segments_is_tuple_not_list(self):
        """Hashable + inmutable: Transcript es value object."""
        t = self._transcript()
        assert isinstance(t.segments, tuple)

    def test_immutability(self):
        t = self._transcript()
        with pytest.raises(Exception):
            t.audio_path = "otro.wav"  # type: ignore[misc]

    def test_empty_transcript_has_no_tags(self):
        from speechlib.domain.transcript import Transcript

        t = Transcript(segments=(), audio_path="x.wav", language="es")
        assert t.diarization_tags == frozenset()

    def test_unidentified_segment_label_is_diarization_tag(self):
        """Test integrador del invariante en el aggregate completo."""
        t = self._transcript()
        labels = [s.speaker.label for s in t.segments]
        assert labels == ["SPEAKER_00", "Manuel", "SPEAKER_00"]
        assert "unknown" not in labels
