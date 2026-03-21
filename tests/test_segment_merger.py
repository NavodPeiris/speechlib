"""Tests: merge_short_turns fusiona turnos cortos del mismo locutor."""
import pytest
from speechlib.segment_merger import merge_short_turns


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
