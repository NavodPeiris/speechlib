"""
Slice 10 AT: relabel_vtt --all-speakers re-evalúa todos los bloques, no solo [unknown].
"""
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import torchaudio


def _make_wav(path: Path, duration_s: float = 2.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _write_vtt(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def _run_relabel(vtt_path, audio_path, voices_folder, extra_args=""):
    """Invoca relabel_vtt.main() con los args dados."""
    import sys
    from speechlib.tools import relabel_vtt as m

    argv = [
        "relabel_vtt",
        str(vtt_path),
        str(audio_path),
        str(voices_folder),
    ] + (extra_args.split() if extra_args else [])

    with patch.object(sys, "argv", argv):
        m.main()


# ── Comportamiento existente preservado (regresión) ─────────────────────────

def test_relabel_vtt_without_flag_only_processes_unknown_blocks(tmp_path):
    """Sin --all-speakers, solo se procesan bloques [unknown] (regresión)."""
    vtt = _write_vtt(
        tmp_path / "test.vtt",
        """\
        WEBVTT

        1
        00:00:00.000 --> 00:00:02.000
        [Agustin] Hola mundo

        2
        00:00:02.000 --> 00:00:04.000
        [unknown] Texto desconocido
        """,
    )
    audio = _make_wav(tmp_path / "audio.wav", duration_s=5.0)
    voices_folder = tmp_path / "voices"
    voices_folder.mkdir()

    fake_emb = np.ones(192)
    embs = {"Agustin": fake_emb, "Juan": fake_emb}

    processed_blocks = []

    def fake_get_embedding(path):
        processed_blocks.append(path)
        return fake_emb

    def fake_find_best(emb, embs, threshold):
        return "Juan"

    with (
        patch("speechlib.tools.relabel_vtt.load_avg_voice_embeddings", return_value=embs),
        patch("speechlib.tools.relabel_vtt.get_embedding", side_effect=fake_get_embedding),
        patch("speechlib.tools.relabel_vtt.find_best_speaker", side_effect=fake_find_best),
        patch("speechlib.tools.relabel_vtt.slice_and_save"),
    ):
        _run_relabel(vtt, audio, voices_folder)

    # Solo se procesó 1 bloque (el [unknown]), no el [Agustin]
    assert len(processed_blocks) == 1

    out = vtt.with_stem(vtt.stem + "_relabeled")
    content = out.read_text(encoding="utf-8")
    assert "[Agustin]" in content   # bloque Agustin intacto
    assert "[unknown]" not in content  # unknown fue relabeled


# ── Con --all-speakers: se procesan todos los bloques ───────────────────────

def test_relabel_vtt_all_speakers_processes_every_block(tmp_path):
    """Con --all-speakers, se calcula embedding para cada bloque del VTT."""
    vtt = _write_vtt(
        tmp_path / "test.vtt",
        """\
        WEBVTT

        1
        00:00:00.000 --> 00:00:02.000
        [Agustin] Hola mundo

        2
        00:00:02.000 --> 00:00:04.000
        [Manuel] Otra cosa
        """,
    )
    audio = _make_wav(tmp_path / "audio.wav", duration_s=5.0)
    voices_folder = tmp_path / "voices"
    voices_folder.mkdir()

    fake_emb = np.ones(192)
    embs = {"Agustin": fake_emb, "Manuel": fake_emb}
    processed_blocks = []

    def fake_get_embedding(path):
        processed_blocks.append(path)
        return fake_emb

    with (
        patch("speechlib.tools.relabel_vtt.load_avg_voice_embeddings", return_value=embs),
        patch("speechlib.tools.relabel_vtt.get_embedding", side_effect=fake_get_embedding),
        patch("speechlib.tools.relabel_vtt.find_best_speaker", return_value="Agustin"),
        patch("speechlib.tools.relabel_vtt.slice_and_save"),
    ):
        _run_relabel(vtt, audio, voices_folder, "--all-speakers")

    # Se procesaron los 2 bloques
    assert len(processed_blocks) == 2


def test_relabel_vtt_all_speakers_corrects_misidentified_block(tmp_path):
    """Bloque [Agustin] donde audio no supera threshold → se corrige a 'unknown'."""
    vtt = _write_vtt(
        tmp_path / "test.vtt",
        """\
        WEBVTT

        1
        00:00:00.000 --> 00:00:02.000
        [Agustin] Texto mal identificado
        """,
    )
    audio = _make_wav(tmp_path / "audio.wav", duration_s=3.0)
    voices_folder = tmp_path / "voices"
    voices_folder.mkdir()

    fake_emb = np.ones(192)
    embs = {"Agustin": fake_emb}

    with (
        patch("speechlib.tools.relabel_vtt.load_avg_voice_embeddings", return_value=embs),
        patch("speechlib.tools.relabel_vtt.get_embedding", return_value=fake_emb),
        # find_best_speaker retorna "unknown" → el bloque se debe reetiqueta
        patch("speechlib.tools.relabel_vtt.find_best_speaker", return_value="unknown"),
        patch("speechlib.tools.relabel_vtt.slice_and_save"),
    ):
        _run_relabel(vtt, audio, voices_folder, "--all-speakers")

    out = vtt.with_stem(vtt.stem + "_relabeled")
    content = out.read_text(encoding="utf-8")
    assert "[Agustin]" not in content
    assert "[unknown]" in content


def test_relabel_vtt_all_speakers_leaves_confirmed_block_unchanged(tmp_path):
    """Bloque [Agustin] donde audio SÍ supera threshold → se mantiene como [Agustin]."""
    vtt = _write_vtt(
        tmp_path / "test.vtt",
        """\
        WEBVTT

        1
        00:00:00.000 --> 00:00:02.000
        [Agustin] Confirmado
        """,
    )
    audio = _make_wav(tmp_path / "audio.wav", duration_s=3.0)
    voices_folder = tmp_path / "voices"
    voices_folder.mkdir()

    fake_emb = np.ones(192)
    embs = {"Agustin": fake_emb}

    with (
        patch("speechlib.tools.relabel_vtt.load_avg_voice_embeddings", return_value=embs),
        patch("speechlib.tools.relabel_vtt.get_embedding", return_value=fake_emb),
        patch("speechlib.tools.relabel_vtt.find_best_speaker", return_value="Agustin"),
        patch("speechlib.tools.relabel_vtt.slice_and_save"),
    ):
        _run_relabel(vtt, audio, voices_folder, "--all-speakers")

    out = vtt.with_stem(vtt.stem + "_relabeled")
    content = out.read_text(encoding="utf-8")
    assert "[Agustin]" in content
