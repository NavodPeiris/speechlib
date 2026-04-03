"""
AT: revert_speakers — replace specified speakers with unknown_NNN by first-appearance order.

Use case: VTT contains false-positive speaker labels from a bad speaker_recognition
run. This tool reverts them to anonymous unknown_001, unknown_002, etc.
"""
from pathlib import Path
import pytest


def _write_vtt(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


SIMPLE_VTT = """\
WEBVTT

1
00:00:00.000 --> 00:00:05.000
[Jolyon] Hello everyone.

2
00:00:05.000 --> 00:00:10.000
[Agustin] Good morning.

3
00:00:10.000 --> 00:00:15.000
[Patricio Renner] Let's start.

4
00:00:15.000 --> 00:00:20.000
[Jolyon] Yes, let's begin.

5
00:00:20.000 --> 00:00:25.000
[Agustin] Sure.

6
00:00:25.000 --> 00:00:30.000
[Francisco] I agree.
"""


def test_reverted_speakers_replaced_by_unknown_NNN(tmp_path):
    """Speakers in the revert list are replaced; others unchanged."""
    from speechlib.tools.revert_speakers import revert_speakers

    vtt = tmp_path / "transcript_es.vtt"
    _write_vtt(vtt, SIMPLE_VTT)

    out = revert_speakers(vtt, ["Jolyon", "Patricio Renner", "Francisco"])

    text = out.read_text(encoding="utf-8")
    assert "[unknown_001]" in text
    assert "[unknown_002]" in text
    assert "[unknown_003]" in text
    assert "[Jolyon]" not in text
    assert "[Patricio Renner]" not in text
    assert "[Francisco]" not in text
    # Non-reverted speaker unchanged
    assert "[Agustin]" in text


def test_unknown_NNN_assigned_by_first_appearance_order(tmp_path):
    """Order of unknown_NNN follows order of first appearance in VTT, not argument order."""
    from speechlib.tools.revert_speakers import revert_speakers

    vtt = tmp_path / "transcript_es.vtt"
    _write_vtt(vtt, SIMPLE_VTT)

    # Jolyon appears first (seg 1), Patricio Renner second (seg 3), Francisco third (seg 6)
    # Even though arg order is different, NNN should follow VTT order
    out = revert_speakers(vtt, ["Francisco", "Patricio Renner", "Jolyon"])

    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Find which label corresponds to which speaker by checking position
    first_jolyon_line = next(i for i, l in enumerate(lines) if "[unknown_" in l and "001" in l or "[unknown_001]" in l)
    # Segment 1 has Jolyon → should become unknown_001
    jolyon_idx = next(i for i, l in enumerate(lines) if "[unknown_001]" in l)
    patricio_idx = next(i for i, l in enumerate(lines) if "[unknown_002]" in l)
    francisco_idx = next(i for i, l in enumerate(lines) if "[unknown_003]" in l)
    assert jolyon_idx < patricio_idx < francisco_idx


def test_output_file_named_with_reverted_suffix(tmp_path):
    """Output file is {stem}_reverted.vtt in the same directory."""
    from speechlib.tools.revert_speakers import revert_speakers

    vtt = tmp_path / "transcript_es.vtt"
    _write_vtt(vtt, SIMPLE_VTT)

    out = revert_speakers(vtt, ["Jolyon"])

    assert out.name == "transcript_es_reverted.vtt"
    assert out.parent == tmp_path
    assert out.exists()


def test_original_file_not_modified(tmp_path):
    """The original VTT is not overwritten."""
    from speechlib.tools.revert_speakers import revert_speakers

    vtt = tmp_path / "transcript_es.vtt"
    _write_vtt(vtt, SIMPLE_VTT)
    original_text = vtt.read_text(encoding="utf-8")

    revert_speakers(vtt, ["Jolyon"])

    assert vtt.read_text(encoding="utf-8") == original_text


def test_speaker_not_in_vtt_ignored(tmp_path):
    """Speakers listed for revert that don't appear in VTT are silently ignored."""
    from speechlib.tools.revert_speakers import revert_speakers

    vtt = tmp_path / "transcript_es.vtt"
    _write_vtt(vtt, SIMPLE_VTT)

    # "Ina Gonzalez" not in VTT
    out = revert_speakers(vtt, ["Jolyon", "Ina Gonzalez"])

    text = out.read_text(encoding="utf-8")
    assert "[unknown_001]" in text
    assert "[unknown_002]" not in text
    assert "[Jolyon]" not in text


def test_all_occurrences_of_speaker_replaced(tmp_path):
    """All segments for a speaker are replaced, not just the first."""
    from speechlib.tools.revert_speakers import revert_speakers

    vtt = tmp_path / "transcript_es.vtt"
    _write_vtt(vtt, SIMPLE_VTT)

    out = revert_speakers(vtt, ["Jolyon"])

    text = out.read_text(encoding="utf-8")
    # Jolyon appears in segments 1 and 4 → both replaced
    assert text.count("[unknown_001]") == 2
    assert "[Jolyon]" not in text
