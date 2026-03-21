"""Tests: write_log_file soporta formato SRT con timestamps SMPTE."""
import os
import pytest
from speechlib.write_log_file import write_log_file, _format_smpte


# ── _format_smpte ──────────────────────────────────────────────────────────────

def test_format_smpte_zero():
    assert _format_smpte(0.0) == "00:00:00,000"

def test_format_smpte_basic():
    assert _format_smpte(125.5) == "00:02:05,500"

def test_format_smpte_over_one_hour():
    assert _format_smpte(3661.75) == "01:01:01,750"

def test_format_smpte_millisecond_precision():
    assert _format_smpte(1.001) == "00:00:01,001"


# ── write_log_file txt (retrocompat) ──────────────────────────────────────────

def test_default_creates_txt_file(tmp_path):
    segments = [[0.0, 1.0, "hello world", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en")
    txt_files = list(tmp_path.glob("*.txt"))
    assert len(txt_files) == 1

def test_txt_format_explicit(tmp_path):
    segments = [[0.0, 1.0, "hello", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="txt")
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert len(list(tmp_path.glob("*.srt"))) == 0


# ── write_log_file srt ────────────────────────────────────────────────────────

def test_srt_format_creates_srt_file(tmp_path):
    segments = [[0.0, 1.5, "hello world", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="srt")
    srt_files = list(tmp_path.glob("*.srt"))
    assert len(srt_files) == 1
    assert len(list(tmp_path.glob("*.txt"))) == 0

def test_srt_block_structure(tmp_path):
    """Cada bloque: número, línea de timestamps, texto, línea vacía."""
    segments = [
        [0.0, 2.0, "hello", "SPEAKER_00"],
        [3.0, 5.0, "world", "SPEAKER_01"],
    ]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="srt")
    content = list(tmp_path.glob("*.srt"))[0].read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    # Bloque 1
    assert lines[0] == "1"
    assert "-->" in lines[1]
    assert "SPEAKER_00" in lines[2]
    assert "hello" in lines[2]
    # Separador
    assert lines[3] == ""
    # Bloque 2
    assert lines[4] == "2"

def test_srt_timestamps_are_smpte(tmp_path):
    segments = [[125.5, 130.0, "text", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="srt")
    content = list(tmp_path.glob("*.srt"))[0].read_text(encoding="utf-8")
    assert "00:02:05,500" in content
    assert "00:02:10,000" in content

def test_srt_skips_empty_text(tmp_path):
    segments = [[0.0, 1.0, "", "SPEAKER_00"], [1.0, 2.0, "real", "SPEAKER_01"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="srt")
    content = list(tmp_path.glob("*.srt"))[0].read_text(encoding="utf-8")
    assert content.strip().startswith("1")  # solo un bloque
