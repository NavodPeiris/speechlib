"""
Slice 4 AT (planner): plan_speaker_samples agrupa los segmentos del
Transcript por SpeakerIdentity.label, selecciona top-N por duracion para
cada speaker (identificados Y no identificados) y filtra los que no llegan
a la duracion minima.

Test puro: solo value objects de dominio. Sin I/O, sin mocks.
"""


def _segment(start_ms, end_ms, label, identified=False):
    from speechlib.domain.transcript import SpeakerIdentity, TranscriptSegment

    if identified:
        speaker = SpeakerIdentity(
            diarization_tag="SPEAKER_00",  # tag arbitrario
            recognized_name=label,
            similarity=0.7,
        )
    else:
        speaker = SpeakerIdentity(diarization_tag=label)

    return TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text="x", speaker=speaker)


def test_planner_creates_one_plan_per_speaker_with_top_n_clips_by_duration():
    from speechlib.domain.transcript import Transcript
    from speechlib.domain.sample_extraction import plan_speaker_samples

    transcript = Transcript(
        segments=(
            # Manuel: 4 segmentos de distintas duraciones
            _segment(0,    1000, "Manuel", identified=True),  # 1000ms
            _segment(2000, 7000, "Manuel", identified=True),  # 5000ms
            _segment(8000, 8500, "Manuel", identified=True),  # 500ms (filtrado por min)
            _segment(9000, 12000, "Manuel", identified=True), # 3000ms
            # SPEAKER_03: 2 segmentos largos
            _segment(13000, 16000, "SPEAKER_03"),  # 3000ms
            _segment(17000, 19500, "SPEAKER_03"),  # 2500ms
        ),
        audio_path="rec.wav",
        language="es",
    )

    plans = plan_speaker_samples(
        transcript,
        max_clips_per_speaker=2,
        min_clip_duration_ms=1000,
    )

    plans_by_label = {p.speaker_label: p for p in plans}

    # Manuel: top-2 por duracion (5000ms y 3000ms), el de 500ms filtrado
    assert "Manuel" in plans_by_label
    manuel = plans_by_label["Manuel"]
    assert manuel.is_identified is True
    assert len(manuel.clips) == 2
    durations = sorted((c.end_ms - c.start_ms for c in manuel.clips), reverse=True)
    assert durations == [5000, 3000]

    # SPEAKER_03: 2 clips, no identificado
    assert "SPEAKER_03" in plans_by_label
    spk03 = plans_by_label["SPEAKER_03"]
    assert spk03.is_identified is False
    assert len(spk03.clips) == 2


def test_planner_includes_both_identified_and_unidentified_speakers():
    """Reemplaza extract_unknown_speakers: ahora cubre TODOS los speakers."""
    from speechlib.domain.transcript import Transcript
    from speechlib.domain.sample_extraction import plan_speaker_samples

    transcript = Transcript(
        segments=(
            _segment(0,    3000, "Pamela", identified=True),
            _segment(4000, 7000, "SPEAKER_05"),
        ),
        audio_path="rec.wav",
        language="es",
    )

    plans = plan_speaker_samples(
        transcript,
        max_clips_per_speaker=5,
        min_clip_duration_ms=1000,
    )

    labels = {p.speaker_label for p in plans}
    assert labels == {"Pamela", "SPEAKER_05"}

    pamela = next(p for p in plans if p.speaker_label == "Pamela")
    spk05 = next(p for p in plans if p.speaker_label == "SPEAKER_05")
    assert pamela.is_identified is True
    assert spk05.is_identified is False
