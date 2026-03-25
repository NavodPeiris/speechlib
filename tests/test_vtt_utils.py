"""AT: speechlib.vtt_utils — parse, write, timestamp conversion."""
from pathlib import Path
import pytest


SIMPLE_VTT = """\
WEBVTT

1
00:00:00.000 --> 00:00:05.000
[Alice] Hello.

2
00:00:05.000 --> 00:00:10.500
[Bob] World.
"""


def test_ts_to_ms_known_value():
    from speechlib.vtt_utils import ts_to_ms
    assert ts_to_ms("01:23:45.678") == 5025678


def test_ts_to_ms_zero():
    from speechlib.vtt_utils import ts_to_ms
    assert ts_to_ms("00:00:00.000") == 0


def test_seconds_to_vtt_ts_known_value():
    from speechlib.vtt_utils import seconds_to_vtt_ts
    assert seconds_to_vtt_ts(5025.678) == "01:23:45.678"


def test_seconds_to_vtt_ts_zero():
    from speechlib.vtt_utils import seconds_to_vtt_ts
    assert seconds_to_vtt_ts(0.0) == "00:00:00.000"


def test_seconds_to_vtt_ts_sub_second():
    from speechlib.vtt_utils import seconds_to_vtt_ts
    assert seconds_to_vtt_ts(0.5) == "00:00:00.500"


def test_roundtrip_parse_write(tmp_path):
    from speechlib.vtt_utils import parse_vtt, write_vtt
    vtt = tmp_path / "test.vtt"
    vtt.write_text(SIMPLE_VTT, encoding="utf-8")

    header, blocks = parse_vtt(vtt)
    out = tmp_path / "out.vtt"
    write_vtt(out, header, blocks)
    _, blocks2 = parse_vtt(out)

    assert len(blocks) == len(blocks2)
    for a, b in zip(blocks, blocks2):
        assert a.speaker == b.speaker
        assert a.text == b.text
        assert a.start_ms == b.start_ms
        assert a.end_ms == b.end_ms


def test_parse_vtt_speaker_and_text():
    from speechlib.vtt_utils import parse_vtt, VttBlock
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(SIMPLE_VTT)
        path = Path(f.name)
    _, blocks = parse_vtt(path)
    assert blocks[0].speaker == "Alice"
    assert blocks[0].text == "Hello."
    assert blocks[1].speaker == "Bob"
    assert blocks[1].start_ms == 5000
    assert blocks[1].end_ms == 10500
    path.unlink()
