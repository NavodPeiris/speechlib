"""
Slice 4 unit tests: plan_speaker_samples (puro).

Tests funcionales sobre value objects. Sin I/O, sin mocks.
"""

import pytest


def _seg(start_ms, end_ms, label, identified=False):
    from speechlib.domain.transcript import SpeakerIdentity, TranscriptSegment

    if identified:
        spk = SpeakerIdentity(
            diarization_tag="SPEAKER_99",
            recognized_name=label,
            similarity=0.8,
        )
    else:
        spk = SpeakerIdentity(diarization_tag=label)
    return TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text="t", speaker=spk)


def _transcript(*segments):
    from speechlib.domain.transcript import Transcript

    return Transcript(
        segments=tuple(segments),
        audio_path="x.wav",
        language="es",
    )


# ── SampleClip / SpeakerSamplePlan value objects ─────────────────────────────


class TestValueObjects:
    def test_sample_clip_is_immutable(self):
        from speechlib.domain.sample_extraction import SampleClip

        clip = SampleClip(start_ms=0, end_ms=1000)
        with pytest.raises(Exception):
            clip.start_ms = 500  # type: ignore[misc]

    def test_sample_clip_value_equality(self):
        from speechlib.domain.sample_extraction import SampleClip

        a = SampleClip(start_ms=0, end_ms=1000)
        b = SampleClip(start_ms=0, end_ms=1000)
        assert a == b

    def test_sample_clip_duration(self):
        from speechlib.domain.sample_extraction import SampleClip

        assert SampleClip(start_ms=500, end_ms=2500).duration_ms == 2000

    def test_speaker_sample_plan_is_immutable(self):
        from speechlib.domain.sample_extraction import SampleClip, SpeakerSamplePlan

        plan = SpeakerSamplePlan(
            speaker_label="X",
            is_identified=True,
            clips=(SampleClip(0, 1000),),
        )
        with pytest.raises(Exception):
            plan.speaker_label = "Y"  # type: ignore[misc]


# ── Agrupacion ───────────────────────────────────────────────────────────────


class TestGrouping:
    def test_segments_with_same_label_grouped_into_single_plan(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(0,    2000, "Manuel", identified=True),
            _seg(3000, 5000, "Manuel", identified=True),
            _seg(6000, 8000, "Manuel", identified=True),
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=5, min_clip_duration_ms=1000
        )

        assert len(plans) == 1
        assert plans[0].speaker_label == "Manuel"
        assert len(plans[0].clips) == 3

    def test_unidentified_speakers_grouped_by_diarization_tag(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(0,    2000, "SPEAKER_00"),
            _seg(3000, 5000, "SPEAKER_00"),
            _seg(6000, 8000, "SPEAKER_01"),
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=5, min_clip_duration_ms=1000
        )

        labels = {p.speaker_label for p in plans}
        assert labels == {"SPEAKER_00", "SPEAKER_01"}

    def test_identified_segments_with_same_name_but_different_tags_collapse(self):
        """Si distintos SPEAKER_XX se reconocen como mismo nombre, se agrupan."""
        from speechlib.domain.sample_extraction import plan_speaker_samples
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )

        transcript = Transcript(
            segments=(
                TranscriptSegment(
                    0, 2000, "x",
                    SpeakerIdentity("SPEAKER_03", "Manuel", 0.8),
                ),
                TranscriptSegment(
                    3000, 5000, "x",
                    SpeakerIdentity("SPEAKER_07", "Manuel", 0.6),
                ),
            ),
            audio_path="a.wav",
            language="es",
        )

        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=5, min_clip_duration_ms=1000
        )
        assert len(plans) == 1
        assert plans[0].speaker_label == "Manuel"
        assert len(plans[0].clips) == 2


# ── Top-N por duracion ───────────────────────────────────────────────────────


class TestTopN:
    def test_picks_top_n_clips_by_duration(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(0,    1000, "X", identified=True),  # 1000
            _seg(2000, 7000, "X", identified=True),  # 5000
            _seg(8000, 10000, "X", identified=True), # 2000
            _seg(11000, 14000, "X", identified=True),# 3000
            _seg(15000, 19000, "X", identified=True),# 4000
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=3, min_clip_duration_ms=500
        )

        assert len(plans[0].clips) == 3
        durations = sorted(
            (c.end_ms - c.start_ms for c in plans[0].clips), reverse=True
        )
        assert durations == [5000, 4000, 3000]

    def test_max_clips_zero_returns_empty_clips(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(0, 5000, "X", identified=True),
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=0, min_clip_duration_ms=1000
        )
        # No deberian existir planes vacios
        assert plans == ()


# ── Filtro de duracion minima ────────────────────────────────────────────────


class TestMinDurationFilter:
    def test_clips_below_min_duration_excluded(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(0,   500,  "X", identified=True),  # 500ms - filtrado
            _seg(1000, 1300, "X", identified=True), # 300ms - filtrado
            _seg(2000, 5000, "X", identified=True), # 3000ms - OK
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=10, min_clip_duration_ms=1000
        )
        assert len(plans[0].clips) == 1
        clip = plans[0].clips[0]
        assert clip.start_ms == 2000
        assert clip.end_ms == 5000

    def test_speaker_with_no_clips_after_filter_excluded_from_plans(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(0,   500,  "X", identified=True),   # filtrado
            _seg(1000, 1200, "X", identified=True),  # filtrado
            _seg(2000, 5000, "Y", identified=True),  # OK
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=5, min_clip_duration_ms=1000
        )
        labels = {p.speaker_label for p in plans}
        assert labels == {"Y"}  # X eliminado completamente


# ── Orden y consistencia ─────────────────────────────────────────────────────


class TestOrdering:
    def test_clips_within_plan_sorted_by_start_ms(self):
        """Aunque la seleccion sea top-N por duracion, los clips finales se
        ordenan por start_ms para consistencia y debugging."""
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(10000, 13000, "X", identified=True),  # 3000ms
            _seg(0,     5000,  "X", identified=True),  # 5000ms
            _seg(20000, 24000, "X", identified=True),  # 4000ms
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=3, min_clip_duration_ms=1000
        )
        starts = [c.start_ms for c in plans[0].clips]
        assert starts == sorted(starts)

    def test_plans_sorted_identified_first_then_unidentified(self):
        """Plans con identified=True primero, luego unidentified.
        Dentro de cada grupo, orden alfabetico estable."""
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(0,    3000, "SPEAKER_05"),
            _seg(4000, 7000, "Manuel", identified=True),
            _seg(8000, 11000, "SPEAKER_02"),
            _seg(12000, 15000, "Agustin", identified=True),
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=5, min_clip_duration_ms=1000
        )
        labels = [p.speaker_label for p in plans]
        # Identified primero (alfabetico), luego unidentified (alfabetico)
        assert labels == ["Agustin", "Manuel", "SPEAKER_02", "SPEAKER_05"]


class TestEdgeCases:
    def test_empty_transcript_returns_empty_tuple(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples
        from speechlib.domain.transcript import Transcript

        transcript = Transcript(segments=(), audio_path="x.wav", language="es")
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=5, min_clip_duration_ms=1000
        )
        assert plans == ()

    def test_returns_tuple_not_list(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(_seg(0, 5000, "X", identified=True))
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=1, min_clip_duration_ms=1000
        )
        assert isinstance(plans, tuple)
        assert isinstance(plans[0].clips, tuple)

    def test_min_duration_zero_includes_all_segments(self):
        from speechlib.domain.sample_extraction import plan_speaker_samples

        transcript = _transcript(
            _seg(0, 100, "X", identified=True),
            _seg(200, 250, "X", identified=True),
        )
        plans = plan_speaker_samples(
            transcript, max_clips_per_speaker=10, min_clip_duration_ms=0
        )
        assert len(plans[0].clips) == 2
