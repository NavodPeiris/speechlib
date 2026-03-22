"""Tests: merge_short_turns, merge_transcript_turns, group_by_sentences, group_by_speaker."""
import pytest
from speechlib.segment_merger import merge_short_turns, group_by_sentences, group_by_speaker


def test_merges_same_speaker_small_gap():
    common = [[0.0, 1.0, "SPEAKER_00"], [1.3, 2.0, "SPEAKER_00"]]
    result = merge_short_turns(common, max_gap_s=0.5)
    assert len(result) == 1
    assert result[0] == [0.0, 2.0, "SPEAKER_00"]


def test_does_not_merge_different_speakers():
    common = [[0.0, 1.0, "SPEAKER_00"], [1.1, 2.0, "SPEAKER_01"]]
    result = merge_short_turns(common, max_gap_s=0.5)
    assert len(result) == 2


def test_does_not_merge_gap_at_threshold():
    """Gap exactamente igual al umbral: NO fusionar."""
    common = [[0.0, 1.0, "SPEAKER_00"], [1.5, 2.5, "SPEAKER_00"]]
    result = merge_short_turns(common, max_gap_s=0.5)
    assert len(result) == 2


def test_does_not_merge_large_gap():
    common = [[0.0, 1.0, "SPEAKER_00"], [2.0, 3.0, "SPEAKER_00"]]
    result = merge_short_turns(common, max_gap_s=0.5)
    assert len(result) == 2


def test_merged_segment_preserves_times():
    common = [[5.0, 6.0, "SPEAKER_00"], [6.2, 7.5, "SPEAKER_00"]]
    result = merge_short_turns(common, max_gap_s=0.5)
    assert result[0][0] == 5.0
    assert result[0][1] == 7.5


def test_chain_merge_three_turns():
    """A→A→A con gaps pequeños: se fusionan en uno solo."""
    common = [
        [0.0, 1.0, "SPEAKER_00"],
        [1.2, 2.0, "SPEAKER_00"],
        [2.3, 3.0, "SPEAKER_00"],
    ]
    result = merge_short_turns(common, max_gap_s=0.5)
    assert len(result) == 1
    assert result[0] == [0.0, 3.0, "SPEAKER_00"]


def test_empty_list():
    assert merge_short_turns([]) == []


def test_single_segment():
    common = [[0.0, 1.0, "SPEAKER_00"]]
    assert merge_short_turns(common) == [[0.0, 1.0, "SPEAKER_00"]]


# ---------------------------------------------------------------------------
# group_by_sentences
# ---------------------------------------------------------------------------

class TestGroupBySentences:
    """group_by_sentences divide bloques largos en segmentos por puntuación."""

    def test_splits_at_period(self):
        """Dos segmentos del mismo speaker: el primero termina en punto → flush."""
        segs = [
            [0.0, 3.0, "Hola mundo.", "Manuel"],
            [3.5, 6.0, "Como están.", "Manuel"],
        ]
        result = group_by_sentences(segs)
        assert len(result) == 2
        assert result[0][2] == "Hola mundo."
        assert result[1][2] == "Como están."

    def test_accumulates_mid_sentence(self):
        """Tres segmentos sin punto final → se fusionan en uno."""
        segs = [
            [0.0, 2.0, "Hola", "Manuel"],
            [2.5, 4.0, "mundo", "Manuel"],
            [4.5, 6.0, "querido", "Manuel"],
        ]
        result = group_by_sentences(segs)
        assert len(result) == 1
        assert "Hola" in result[0][2]
        assert "querido" in result[0][2]

    def test_speaker_change_always_flushes(self):
        """Cambio de speaker fuerza flush aunque no haya punto."""
        segs = [
            [0.0, 3.0, "Hola", "Manuel"],
            [3.5, 6.0, "mundo", "Francisco"],
        ]
        result = group_by_sentences(segs)
        assert len(result) == 2
        assert result[0][3] == "Manuel"
        assert result[1][3] == "Francisco"

    def test_max_chars_at_strong_pause(self):
        """Superar max_chars con texto que termina en coma → flush."""
        long_text = "a" * 85 + ","
        segs = [
            [0.0, 5.0, long_text, "Agustin"],
            [5.5, 8.0, "siguiente frase.", "Agustin"],
        ]
        result = group_by_sentences(segs, max_chars=80)
        # El segmento largo debe haber sido flusheado solo
        assert result[0][2] == long_text

    def test_sentence_end_patterns(self):
        """Signos ! ? ; también cierran sentencia."""
        for punct in ["!", "?", ";"]:
            segs = [
                [0.0, 2.0, f"texto{punct}", "X"],
                [2.5, 4.0, "nuevo.", "X"],
            ]
            result = group_by_sentences(segs)
            assert len(result) == 2, f"Fallo con signo '{punct}'"

    def test_timestamps_preserved(self):
        """El segmento fusionado tiene start del primero y end del último."""
        segs = [
            [1.0, 3.0, "primera", "A"],
            [3.5, 5.0, "segunda.", "A"],
        ]
        result = group_by_sentences(segs)
        # Ambos se acumulan hasta el punto final del segundo
        fused = result[0]
        assert fused[0] == 1.0
        assert fused[1] == 5.0

    def test_empty_list(self):
        assert group_by_sentences([]) == []

    def test_single_segment(self):
        seg = [0.0, 3.0, "Hola mundo.", "A"]
        assert group_by_sentences([seg]) == [[0.0, 3.0, "Hola mundo.", "A"]]


# ---------------------------------------------------------------------------
# group_by_speaker
# ---------------------------------------------------------------------------

class TestGroupBySpeaker:
    """group_by_speaker fusiona solo por cambio de speaker, sin límite de gap."""

    def test_merges_same_speaker_large_gap(self):
        """Gap grande del mismo speaker → se fusiona."""
        segs = [
            [0.0, 5.0, "primera parte.", "Jolyon"],
            [20.0, 30.0, "segunda parte.", "Jolyon"],
        ]
        result = group_by_speaker(segs)
        assert len(result) == 1
        assert "primera parte." in result[0][2]
        assert "segunda parte." in result[0][2]

    def test_splits_on_speaker_change(self):
        """Speaker distinto en el medio interrumpe la fusión."""
        segs = [
            [0.0, 5.0, "hola.", "Jolyon"],
            [5.5, 8.0, "interrumpe.", "Agustin"],
            [8.5, 12.0, "continúa.", "Jolyon"],
        ]
        result = group_by_speaker(segs)
        jolyon_segs = [s for s in result if s[3] == "Jolyon"]
        assert len(jolyon_segs) == 2
