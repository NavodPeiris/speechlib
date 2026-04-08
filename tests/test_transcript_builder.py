"""
Slice 5 unit tests: build_transcript_from_legacy_segments (puro).
"""

import pytest


def test_overlap_resolves_correct_tag_when_segment_within_turn():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[1.0, 1.5, "x", "Manuel"]]
    annotation = [
        (0.0, 5.0, "SPEAKER_00"),  # cubre completamente el segmento
        (5.0, 10.0, "SPEAKER_01"),
    ]
    speaker_map = {"SPEAKER_00": "Manuel", "SPEAKER_01": "Pamela"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    assert t.segments[0].speaker.diarization_tag == "SPEAKER_00"


def test_overlap_picks_max_overlap_when_segment_spans_two_turns():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[1.0, 4.0, "x", "X"]]
    annotation = [
        (0.0, 1.5, "SPEAKER_00"),  # overlap = 0.5s
        (1.5, 5.0, "SPEAKER_01"),  # overlap = 2.5s ← gana
    ]
    speaker_map = {"SPEAKER_00": "X", "SPEAKER_01": "X"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    assert t.segments[0].speaker.diarization_tag == "SPEAKER_01"


def test_segment_outside_any_turn_falls_back_to_label():
    """Si por algun motivo no hay overlap, el diarization_tag = label."""
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[100.0, 105.0, "x", "SPEAKER_99"]]
    annotation = [(0.0, 10.0, "SPEAKER_00")]
    speaker_map = {"SPEAKER_00": "X"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    assert t.segments[0].speaker.diarization_tag == "SPEAKER_99"
    assert t.segments[0].speaker.label == "SPEAKER_99"


def test_label_unknown_string_normalized_to_diarization_tag():
    """Si por bug del legacy aparece label='unknown', NO debe propagarse al
    Transcript: el dominio nuevo lo normaliza al diarization_tag pyannote."""
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[1.0, 2.0, "x", "unknown"]]
    annotation = [(0.0, 5.0, "SPEAKER_07")]
    speaker_map = {"SPEAKER_07": "unknown"}  # legacy bug

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    spk = t.segments[0].speaker
    assert spk.diarization_tag == "SPEAKER_07"
    assert spk.recognized_name is None
    assert spk.label == "SPEAKER_07"
    assert spk.label != "unknown"


def test_int_ms_conversion():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[1.234, 5.678, "x", "X"]]
    annotation = [(0.0, 10.0, "SPEAKER_00")]
    speaker_map = {"SPEAKER_00": "X"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    assert t.segments[0].start_ms == 1234
    assert t.segments[0].end_ms == 5678


def test_returns_immutable_transcript():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments
    from speechlib.domain.transcript import Transcript

    t = build_transcript_from_legacy_segments(
        [[0.0, 1.0, "x", "X"]],
        [(0.0, 1.0, "SPEAKER_00")],
        {"SPEAKER_00": "X"},
        "a.wav", "es",
    )
    assert isinstance(t, Transcript)
    assert isinstance(t.segments, tuple)
