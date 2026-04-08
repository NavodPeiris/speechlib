"""
Slice 1 AT: SpeakerIdentity nunca pierde el diarization_tag.

Invariante anti-bug: incluso cuando no hay reconocimiento contra la libreria
de voces, label() debe fallback al SPEAKER_XX original de pyannote — nunca
al literal "unknown". Este es el comportamiento que el bug del relabel_vtt
--all-speakers violaba.

Test puro: solo opera sobre value objects del dominio. Sin I/O, sin mocks.
"""


def test_unidentified_speaker_label_falls_back_to_diarization_tag():
    from speechlib.domain.transcript import SpeakerIdentity

    speaker = SpeakerIdentity(diarization_tag="SPEAKER_05")

    assert speaker.label == "SPEAKER_05"
    assert speaker.is_identified is False


def test_identified_speaker_label_uses_recognized_name():
    from speechlib.domain.transcript import SpeakerIdentity

    speaker = SpeakerIdentity(
        diarization_tag="SPEAKER_05",
        recognized_name="Manuel Olguin",
        similarity=0.72,
    )

    assert speaker.label == "Manuel Olguin"
    assert speaker.is_identified is True


def test_speaker_label_never_returns_literal_unknown_string():
    """Invariante: la palabra 'unknown' jamas debe aparecer como label.
    El fallback siempre es el tag pyannote."""
    from speechlib.domain.transcript import SpeakerIdentity

    for tag in ("SPEAKER_00", "SPEAKER_07", "SPEAKER_15"):
        speaker = SpeakerIdentity(diarization_tag=tag)
        assert speaker.label != "unknown"
        assert speaker.label == tag
