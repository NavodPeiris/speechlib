"""
Slice 3 unit tests: serializacion de Transcript a JSON.

Tests funcionales puros: tmp_path + value objects. Sin mocks.
Verifican formato del JSON y casos limite del round-trip.
"""

import json

import pytest


def _make_transcript():
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )

    return Transcript(
        segments=(
            TranscriptSegment(
                start_ms=0,
                end_ms=1000,
                text="primero",
                speaker=SpeakerIdentity(diarization_tag="SPEAKER_00"),
            ),
            TranscriptSegment(
                start_ms=1000,
                end_ms=2500,
                text="segundo",
                speaker=SpeakerIdentity(
                    diarization_tag="SPEAKER_01",
                    recognized_name="Pamela",
                    similarity=0.55,
                ),
            ),
        ),
        audio_path="audio.wav",
        language="es",
    )


# ── Round-trip ───────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_full_round_trip_equality(self, tmp_path):
        from speechlib.domain.transcript import Transcript

        original = _make_transcript()
        path = tmp_path / "t.json"
        original.save(path)
        loaded = Transcript.load(path)

        assert loaded == original

    def test_empty_transcript_round_trip(self, tmp_path):
        from speechlib.domain.transcript import Transcript

        original = Transcript(segments=(), audio_path="x.wav", language="es")
        path = tmp_path / "empty.json"
        original.save(path)
        loaded = Transcript.load(path)

        assert loaded == original
        assert loaded.segments == ()

    def test_similarity_none_round_trip(self, tmp_path):
        """similarity=None debe sobrevivir el round-trip (no convertirse en 0)."""
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )

        original = Transcript(
            segments=(
                TranscriptSegment(
                    0, 1000, "x",
                    SpeakerIdentity(diarization_tag="SPEAKER_00"),
                ),
            ),
            audio_path="x.wav",
            language="es",
        )
        path = tmp_path / "t.json"
        original.save(path)
        loaded = Transcript.load(path)

        assert loaded.segments[0].speaker.similarity is None
        assert loaded.segments[0].speaker.recognized_name is None

    def test_similarity_float_round_trip_precision(self, tmp_path):
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )

        original = Transcript(
            segments=(
                TranscriptSegment(
                    0, 1000, "x",
                    SpeakerIdentity(
                        diarization_tag="SPEAKER_00",
                        recognized_name="X",
                        similarity=0.123456789,
                    ),
                ),
            ),
            audio_path="x.wav",
            language="es",
        )
        path = tmp_path / "t.json"
        original.save(path)
        loaded = Transcript.load(path)

        assert loaded.segments[0].speaker.similarity == pytest.approx(0.123456789)


# ── Formato del JSON ─────────────────────────────────────────────────────────


class TestJsonFormat:
    def test_json_is_human_readable(self, tmp_path):
        """El JSON debe ser parseable y contener los campos esperados."""
        original = _make_transcript()
        path = tmp_path / "t.json"
        original.save(path)

        data = json.loads(path.read_text(encoding="utf-8"))

        assert "segments" in data
        assert "audio_path" in data
        assert "language" in data
        assert data["audio_path"] == "audio.wav"
        assert data["language"] == "es"
        assert len(data["segments"]) == 2

    def test_segment_has_speaker_with_diarization_tag(self, tmp_path):
        original = _make_transcript()
        path = tmp_path / "t.json"
        original.save(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        seg0 = data["segments"][0]

        assert seg0["start_ms"] == 0
        assert seg0["end_ms"] == 1000
        assert seg0["text"] == "primero"
        assert seg0["speaker"]["diarization_tag"] == "SPEAKER_00"
        assert seg0["speaker"]["recognized_name"] is None
        assert seg0["speaker"]["similarity"] is None

    def test_identified_segment_has_name_and_similarity(self, tmp_path):
        original = _make_transcript()
        path = tmp_path / "t.json"
        original.save(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        seg1 = data["segments"][1]

        assert seg1["speaker"]["recognized_name"] == "Pamela"
        assert seg1["speaker"]["similarity"] == 0.55

    def test_schema_version_is_present(self, tmp_path):
        """Versionar el schema permite migracion futura sin romper archivos."""
        original = _make_transcript()
        path = tmp_path / "t.json"
        original.save(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" in data
        assert isinstance(data["schema_version"], int)

    def test_save_creates_parent_directory_if_missing(self, tmp_path):
        from speechlib.domain.transcript import Transcript

        original = _make_transcript()
        path = tmp_path / "nested" / "subdir" / "t.json"
        original.save(path)

        assert path.exists()
        loaded = Transcript.load(path)
        assert loaded == original


# ── Errores ──────────────────────────────────────────────────────────────────


class TestErrors:
    def test_load_nonexistent_raises(self, tmp_path):
        from speechlib.domain.transcript import Transcript

        with pytest.raises(FileNotFoundError):
            Transcript.load(tmp_path / "missing.json")

    def test_load_unknown_schema_version_raises(self, tmp_path):
        from speechlib.domain.transcript import Transcript

        path = tmp_path / "t.json"
        path.write_text(
            json.dumps({"schema_version": 999, "segments": [], "audio_path": "x", "language": "es"}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="schema"):
            Transcript.load(path)
