"""Tests: write_log_file soporta formato VTT con timestamps WebVTT."""
import os
import pytest
from speechlib.write_log_file import write_log_file, _format_vtt


# ── _format_vtt ───────────────────────────────────────────────────────────────

def test_format_vtt_zero():
    assert _format_vtt(0.0) == "00:00:00.000"

def test_format_vtt_basic():
    assert _format_vtt(125.5) == "00:02:05.500"

def test_format_vtt_over_one_hour():
    assert _format_vtt(3661.75) == "01:01:01.750"

def test_format_vtt_millisecond_precision():
    assert _format_vtt(1.001) == "00:00:01.001"


# ── write_log_file default (vtt) ───────────────────────────────────────────────

def test_default_creates_vtt_file(tmp_path):
    segments = [[0.0, 1.0, "hello world", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en")
    vtt_files = list(tmp_path.glob("*.vtt"))
    assert len(vtt_files) == 1
    assert len(list(tmp_path.glob("*.txt"))) == 0


# ── write_log_file txt (retrocompat) ──────────────────────────────────────────

def test_txt_format_explicit(tmp_path):
    segments = [[0.0, 1.0, "hello", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="txt")
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert len(list(tmp_path.glob("*.vtt"))) == 0


# ── write_log_file vtt ────────────────────────────────────────────────────────

def test_vtt_format_creates_vtt_file(tmp_path):
    segments = [[0.0, 1.5, "hello world", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="vtt")
    vtt_files = list(tmp_path.glob("*.vtt"))
    assert len(vtt_files) == 1
    assert len(list(tmp_path.glob("*.txt"))) == 0

def test_vtt_header_present(tmp_path):
    """El archivo VTT comienza con la línea 'WEBVTT'."""
    segments = [[0.0, 1.0, "texto", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="vtt")
    content = list(tmp_path.glob("*.vtt"))[0].read_text(encoding="utf-8")
    assert content.startswith("WEBVTT"), (
        f"El VTT debe empezar con 'WEBVTT'. Inicio: {content[:50]!r}"
    )

def test_vtt_block_structure(tmp_path):
    """Estructura: WEBVTT, línea vacía, número, timestamp, texto, línea vacía."""
    segments = [
        [0.0, 2.0, "hello", "SPEAKER_00"],
        [3.0, 5.0, "world", "SPEAKER_01"],
    ]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="vtt")
    content = list(tmp_path.glob("*.vtt"))[0].read_text(encoding="utf-8")
    lines = content.split("\n")
    assert lines[0] == "WEBVTT"
    assert lines[1] == ""
    assert lines[2] == "1"
    assert "-->" in lines[3]
    assert "SPEAKER_00" in lines[4]
    assert "hello" in lines[4]
    assert lines[5] == ""
    assert lines[6] == "2"

def test_vtt_timestamps_use_dot(tmp_path):
    """Los timestamps VTT usan punto como separador de milisegundos."""
    segments = [[125.5, 130.0, "text", "SPEAKER_00"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="vtt")
    content = list(tmp_path.glob("*.vtt"))[0].read_text(encoding="utf-8")
    assert "00:02:05.500" in content
    assert "00:02:10.000" in content
    assert "," not in content.split("WEBVTT")[1], "No debe haber comas en timestamps VTT"

def test_vtt_skips_empty_text(tmp_path):
    segments = [[0.0, 1.0, "", "SPEAKER_00"], [1.0, 2.0, "real", "SPEAKER_01"]]
    write_log_file(segments, str(tmp_path), "audio.wav", "en", output_format="vtt")
    content = list(tmp_path.glob("*.vtt"))[0].read_text(encoding="utf-8")
    blocks = [b for b in content.split("\n\n") if b.strip() and b.strip() != "WEBVTT"]
    assert len(blocks) == 1, f"Se esperaba 1 bloque, hay {len(blocks)}"

def test_vtt_overlapping_cues_both_written(tmp_path):
    """Dos cues solapados se escriben ambos — VTT los soporta nativamente."""
    segments = [
        [37.3, 40.4, "Tiene que opinar el público.", "Agustin"],
        [38.3, 41.2, "damos uno de nuevo, por favor.", "Manuel"],
    ]
    write_log_file(segments, str(tmp_path), "audio.wav", "es", output_format="vtt")
    content = list(tmp_path.glob("*.vtt"))[0].read_text(encoding="utf-8")
    assert "Agustin" in content
    assert "Manuel" in content
    assert "00:00:37.300" in content
    assert "00:00:38.300" in content
