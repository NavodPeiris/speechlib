"""
E2E: Validación de las 4 ventajas migradas de wx3 a speechlib.

Corre core_analysis UNA VEZ sobre examples/obama_zach.wav (fixture de sesión)
y valida cada ventaja con assertions claras.

Requisitos:
    - HF_TOKEN env var con acceso a pyannote/speaker-diarization-3.1
    - examples/obama_zach.wav presente en el repo

Uso:
    cd c:/workspace/#dev/speechlib
    HF_TOKEN=hf_... pytest tests/test_e2e_wx3_advantages.py -v -s -m e2e
"""

import os
import re
import time
from pathlib import Path

import pytest

# ── Condiciones de skip ────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
AUDIO = Path(__file__).parent.parent / "examples" / "obama_zach.wav"

pytestmark = pytest.mark.e2e

skip_reason = []
if not HF_TOKEN:
    skip_reason.append("HF_TOKEN no está seteado")
if not AUDIO.exists():
    skip_reason.append(f"audio no encontrado: {AUDIO}")

needs_hf = pytest.mark.skipif(bool(skip_reason), reason=" | ".join(skip_reason) or "ok")


# ── Fixtures de sesión: core_analysis corre UNA sola vez ─────────────────────


@pytest.fixture(scope="session")
def run_result(tmp_path_factory):
    """
    Corre core_analysis sobre obama_zach.wav con output_format='vtt' (default).

    Usa un spy sobre merge_short_turns para capturar segmentos antes/después
    sin necesidad de una corrida extra del pipeline.
    Retorna (common_segments, log_dir, cache_info_after, merge_before, merge_after).
    """
    from unittest.mock import patch
    from speechlib.core_analysis import core_analysis
    from speechlib.transcribe import _get_faster_whisper_model
    from speechlib.segment_merger import merge_short_turns as _real_merge

    _get_faster_whisper_model.cache_clear()

    log_dir = tmp_path_factory.mktemp("e2e_vtt")

    merge_counts = {}

    def spy_merge(common, max_gap_s=0.5):
        result = _real_merge(common, max_gap_s)
        merge_counts["before"] = len(common)
        merge_counts["after"] = len(result)
        return result

    t0 = time.time()
    with patch("speechlib.core_analysis.merge_short_turns", spy_merge):
        result = core_analysis(
            str(AUDIO),
            voices_folder=None,
            log_folder=str(log_dir),
            language="en",
            modelSize="base",
            ACCESS_TOKEN=HF_TOKEN,
            model_type="faster-whisper",
        )
    elapsed = time.time() - t0

    cache_info = _get_faster_whisper_model.cache_info()
    print(f"\n[E2E] core_analysis completado en {elapsed:.1f}s")
    print(f"[E2E] segmentos: {len(result)}, speakers: {len({s[3] for s in result})}")
    print(f"[E2E] cache hits={cache_info.hits}, misses={cache_info.misses}")
    print(f"[E2E] merger: {merge_counts.get('before', '?')} -> {merge_counts.get('after', '?')} segmentos")

    return result, log_dir, cache_info, merge_counts


@pytest.fixture(scope="session")
def run_result_vtt(tmp_path_factory, run_result):
    """
    Segunda corrida con output_format='vtt'. Reutiliza el modelo ya cacheado
    (valida que el cache sobrevive entre llamadas de la misma sesión).
    """
    from speechlib.core_analysis import core_analysis
    from speechlib.transcribe import _get_faster_whisper_model

    log_dir = tmp_path_factory.mktemp("e2e_vtt2")

    cache_before = _get_faster_whisper_model.cache_info()

    result = core_analysis(
        str(AUDIO),
        voices_folder=None,
        log_folder=str(log_dir),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=HF_TOKEN,
        model_type="faster-whisper",
        output_format="vtt",
    )

    cache_after = _get_faster_whisper_model.cache_info()
    new_hits = cache_after.hits - cache_before.hits
    new_misses = cache_after.misses - cache_before.misses
    print(f"\n[E2E VTT] nueva corrida: hits_nuevos={new_hits}, misses_nuevos={new_misses}")

    return result, log_dir, new_hits, new_misses


# ══════════════════════════════════════════════════════════════════════════════
# VENTAJA 1 — Model caching (lru_cache en _get_faster_whisper_model)
# Invariante: WhisperModel se construye 1 vez por configuración, no N veces.
# ══════════════════════════════════════════════════════════════════════════════


@needs_hf
def test_v1_model_constructed_only_once(run_result):
    """WhisperModel se construyó exactamente 1 vez para todos los segmentos."""
    _, _, cache_info, _ = run_result
    assert cache_info.misses == 1, (
        f"Se esperaba 1 construcción del modelo, hubo {cache_info.misses}. "
        "Esto indica que _get_faster_whisper_model no tiene cache o fue llamado "
        "con parámetros distintos."
    )


@needs_hf
def test_v1_cache_hits_consistent_with_transcription_mode(run_result):
    """Con transcribe_full_aligned, el modelo se usa 1 sola vez (miss=1, hits>=0).

    Nota: antes de slice 1 (full-file transcription), hits > 0 porque el modelo
    se reutilizaba N veces (una por segmento). Ahora con una sola llamada a
    transcribe_full_aligned, hits=0 y miss=1 es el resultado esperado.
    En la segunda corrida (VTT), el hit viene del cache entre corridas.
    """
    _, _, cache_info, _ = run_result
    assert cache_info.misses == 1, (
        f"Se esperaba 1 miss (una sola construcción), hubo {cache_info.misses}."
    )


@needs_hf
def test_v1_second_run_uses_only_cache(run_result_vtt):
    """
    La segunda corrida (VTT) no construye un nuevo modelo:
    todos sus accesos van a cache (misses_nuevos == 0).
    """
    _, _, new_hits, new_misses = run_result_vtt
    assert new_misses == 0, (
        f"La segunda corrida construyó {new_misses} instancias nuevas de WhisperModel. "
        "Se esperaba 0 — el modelo de la primera corrida debe sobrevivir entre llamadas."
    )
    assert new_hits > 0, (
        "La segunda corrida no usó el cache. "
        f"new_hits={new_hits}, new_misses={new_misses}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# VENTAJA 2 — pyannote/speaker-diarization-3.1
# Invariante: diarización detecta múltiples locutores y preserva timestamps.
# ══════════════════════════════════════════════════════════════════════════════


@needs_hf
def test_v2_multiple_speakers_detected(run_result):
    """obama_zach.wav tiene 2 locutores — la diarización debe encontrar ≥ 2."""
    result, _, _, _ = run_result
    speakers = {seg[3] for seg in result}
    assert len(speakers) >= 2, (
        f"Se esperaban ≥2 locutores, se encontraron: {speakers}. "
        "Verificar que pyannote/speaker-diarization-3.1 está configurado correctamente."
    )


@needs_hf
def test_v2_timestamps_positive_and_increasing(run_result):
    """Todos los segmentos tienen end >= start >= 0; la mayoría son de duración positiva."""
    result, _, _, _ = run_result
    assert result, "El resultado está vacío"
    for seg in result:
        start, end = seg[0], seg[1]
        assert start >= 0, f"start negativo: {seg}"
        assert end >= start, f"end < start: {seg}"
    # Al menos el 80% de los segmentos tienen duración positiva (algunos son zero-duration de pyannote)
    positive_duration = sum(1 for s in result if s[1] > s[0])
    assert positive_duration / len(result) >= 0.8, (
        f"Demasiados segmentos de duración cero: {len(result) - positive_duration}/{len(result)}"
    )


@needs_hf
def test_v2_first_segment_starts_near_zero(run_result):
    """El primer segmento empieza dentro de los primeros 15 segundos."""
    result, _, _, _ = run_result
    assert result[0][0] < 15.0, (
        f"Primer segmento empieza en {result[0][0]}s — demasiado tarde. "
        "Posible problema con la diarización."
    )


@needs_hf
def test_v2_segment_count_is_reasonable(run_result):
    """obama_zach.wav (2 speakers) debe producir entre 2 y 200 segmentos."""
    result, _, _, _ = run_result
    assert 2 <= len(result) <= 200, (
        f"Número de segmentos inesperado: {len(result)}. "
        "Muy pocos (<2) o fragmentación excesiva (>200)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# VENTAJA 3 — Salida VTT con timestamps WebVTT
# Invariante: output_format='vtt' produce bloques VTT válidos y bien formateados.
# ══════════════════════════════════════════════════════════════════════════════


@needs_hf
def test_v3_vtt_file_created(run_result_vtt):
    """output_format='vtt' crea transcript_en.vtt en artifacts_dir."""
    artifacts_dir = AUDIO.parent / f".{AUDIO.stem}"
    vtt = artifacts_dir / "transcript_en.vtt"
    assert vtt.exists(), f"transcript_en.vtt no encontrado en {artifacts_dir}"
    assert not (artifacts_dir / "transcript_en.txt").exists(), "No debe haber .txt en corrida VTT"


@needs_hf
def test_v3_vtt_starts_with_webvtt_header(run_result_vtt):
    """El archivo VTT comienza con 'WEBVTT'."""
    artifacts_dir = AUDIO.parent / f".{AUDIO.stem}"
    content = (artifacts_dir / "transcript_en.vtt").read_text(encoding="utf-8")
    assert content.startswith("WEBVTT"), (
        f"El VTT no empieza con 'WEBVTT'. Primeros 100 chars:\n{content[:100]}"
    )


@needs_hf
def test_v3_vtt_timestamps_use_dot(run_result_vtt):
    """Los timestamps tienen formato VTT: HH:MM:SS.mmm --> HH:MM:SS.mmm (punto, no coma)."""
    artifacts_dir = AUDIO.parent / f".{AUDIO.stem}"
    content = (artifacts_dir / "transcript_en.vtt").read_text(encoding="utf-8")
    pattern = r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}"
    matches = re.findall(pattern, content)
    assert len(matches) >= 1, (
        f"No se encontraron timestamps VTT. Primeros 300 chars:\n{content[:300]}"
    )
    # Verificar que las líneas de timestamp no usan coma (SRT) sino punto (VTT)
    smpte_pattern = r"\d{2}:\d{2}:\d{2},\d{3}"
    smpte_matches = re.findall(smpte_pattern, content)
    assert len(smpte_matches) == 0, f"Timestamps con coma (formato SRT) encontrados: {smpte_matches[:3]}"


@needs_hf
def test_v3_vtt_block_structure_is_valid(run_result_vtt):
    """Cada bloque VTT tiene: número, línea -->, texto con speaker."""
    artifacts_dir = AUDIO.parent / f".{AUDIO.stem}"
    content = (artifacts_dir / "transcript_en.vtt").read_text(encoding="utf-8")
    blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip() and b.strip() != "WEBVTT"]
    assert len(blocks) >= 1, "No se encontraron bloques VTT"

    for i, block in enumerate(blocks[:5], start=1):
        lines = block.split("\n")
        assert lines[0] == str(i), f"Bloque {i}: primera línea debe ser '{i}', es '{lines[0]}'"
        assert "-->" in lines[1], f"Bloque {i}: segunda línea debe tener '-->', es '{lines[1]}'"
        assert len(lines) >= 3, f"Bloque {i}: falta línea de texto"
        assert "SPEAKER_" in lines[2], (
            f"Bloque {i}: falta etiqueta de locutor en '{lines[2]}'"
        )


@needs_hf
def test_v3_vtt_is_default_format(run_result):
    """La corrida por defecto (sin output_format) crea transcript_en.vtt en artifacts_dir."""
    artifacts_dir = AUDIO.parent / f".{AUDIO.stem}"
    assert (artifacts_dir / "transcript_en.vtt").exists(), "transcript_en.vtt no encontrado"
    assert not (artifacts_dir / "transcript_en.txt").exists(), "No debe haber .txt en corrida por defecto"


# ══════════════════════════════════════════════════════════════════════════════
# VENTAJA 4 — Fusión de turnos cortos (merge_short_turns)
# Invariante: en el resultado final no hay dos segmentos consecutivos del mismo
# locutor con gap < 0.5s (todos fueron fusionados por merge_short_turns).
# ══════════════════════════════════════════════════════════════════════════════


# ── Benchmark de reducción ────────────────────────────────────────────────────

@needs_hf
def test_v4_merger_reduces_segment_count(run_result):
    """
    merge_short_turns debe reducir el número de segmentos respecto a la
    diarización cruda. Evidencia cuantitativa de la mejora.
    """
    _, _, _, merge_counts = run_result
    before = merge_counts["before"]
    after = merge_counts["after"]
    reduction = before - after
    pct = reduction / before * 100 if before else 0

    print(f"\n[Merger benchmark] {before} -> {after} segmentos ({reduction} fusionados, {pct:.1f}% reducción)")

    assert after < before, (
        f"El merger no redujo segmentos: antes={before}, después={after}. "
        "Posiblemente el audio no tiene turnos cortos consecutivos del mismo speaker."
    )


@needs_hf
def test_v4_merger_reduction_is_meaningful(run_result):
    """
    La reducción debe ser de al menos 1 segmento. Si el audio tiene fragmentación
    real, esperamos al menos 5% de reducción.
    """
    _, _, _, merge_counts = run_result
    before = merge_counts["before"]
    after = merge_counts["after"]
    pct = (before - after) / before * 100 if before else 0

    assert pct >= 1.0, (
        f"Reducción muy baja: {pct:.1f}% ({before} -> {after}). "
        "El merger casi no tuvo efecto en este audio."
    )


@needs_hf
def test_v4_no_mergeable_consecutive_segments(run_result):
    """
    No existen pares (seg_i, seg_{i+1}) del mismo locutor con gap < 0.5s.
    Si existieran, merge_short_turns debió haberlos fusionado antes de la transcripción.
    """
    result, _, _, _ = run_result
    violations = []
    for i in range(len(result) - 1):
        curr = result[i]
        nxt = result[i + 1]
        if curr[3] == nxt[3]:  # mismo locutor
            gap = round(nxt[0] - curr[1], 3)
            if gap < 0.5:
                violations.append((curr, nxt, gap))

    assert len(violations) == 0, (
        f"Se encontraron {len(violations)} pares sin fusionar del mismo locutor "
        f"con gap < 0.5s. merge_short_turns no está aplicado correctamente.\n"
        + "\n".join(f"  gap={v[2]:.3f}s: {v[0]} | {v[1]}" for v in violations[:3])
    )


@needs_hf
def test_v4_segments_with_duration_have_text(run_result):
    """
    Segmentos con duración > 0 deben tener texto (pyannote puede devolver
    algunos zero-duration que producen texto vacío — son aceptables).
    """
    result, _, _, _ = run_result
    with_dur = [s for s in result if s[1] > s[0]]
    empty_with_dur = [s for s in with_dur if not s[2].strip()]
    # Al menos el 70% de los segmentos con duración deben tener texto
    if with_dur:
        text_rate = (len(with_dur) - len(empty_with_dur)) / len(with_dur)
        assert text_rate >= 0.7, (
            f"Solo {text_rate:.0%} de segmentos con duración tienen texto. "
            f"Segmentos vacíos con duración: {empty_with_dur[:3]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN: fixture de sesión imprime tabla con los 4 invariantes al final
# ══════════════════════════════════════════════════════════════════════════════


@needs_hf
def test_summary_print(run_result, run_result_vtt):
    """Imprime resumen de los 4 invariantes para inspección manual."""
    result, log_dir_vtt_default, cache_info, merge_counts = run_result
    _, log_dir_vtt2, new_hits_vtt, new_misses_vtt = run_result_vtt

    speakers = {seg[3] for seg in result}
    artifacts_dir = AUDIO.parent / f".{AUDIO.stem}"
    vtt_content = (artifacts_dir / "transcript_en.vtt").read_text(encoding="utf-8")
    vtt_blocks = [b for b in vtt_content.strip().split("\n\n") if b.strip() and b.strip() != "WEBVTT"]

    violations = sum(
        1
        for i in range(len(result) - 1)
        if result[i][3] == result[i + 1][3] and (result[i + 1][0] - result[i][1]) < 0.5
    )

    print("\n")
    print("=" * 60)
    print("  RESUMEN E2E — Ventajas wx3 en speechlib")
    print("=" * 60)
    print(f"  V1 Caching   hits={cache_info.hits}, misses={cache_info.misses} "
          f"| 2ª corrida: hits_nuevos={new_hits_vtt}, misses_nuevos={new_misses_vtt}")
    print(f"  V2 Pyannote  {len(result)} segmentos, {len(speakers)} locutores: {speakers}")
    print(f"  V3 VTT       {len(vtt_blocks)} bloques | "
          f".vtt={artifacts_dir / 'transcript_en.vtt'}")
    before = merge_counts.get("before", "?")
    after = merge_counts.get("after", "?")
    pct = f"{(before - after) / before * 100:.1f}%" if isinstance(before, int) and before else "?"
    print(f"  V4 Merger    {before} -> {after} segmentos ({pct} reducción) | pares sin fusionar: {violations}")
    print("=" * 60)
    assert True  # este test solo imprime, las assertions están en los tests anteriores
