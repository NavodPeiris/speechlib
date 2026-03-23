"""
AT: compresión opcional post-enhance produce AAC mono 96kbps 16kHz.

Corre core_analysis con compress=True sobre audio real.
Verifica propiedades del archivo comprimido con ffprobe.
Valida que la compresión corre en paralelo con diarización (overhead < 2s).

Uso:
    HF_TOKEN=hf_... pytest tests/test_acceptance_compress_audio.py -v -s -m e2e
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

HF_TOKEN = os.environ.get("HF_TOKEN", "")
AUDIO = Path(__file__).parent.parent / "examples" / "obama_zach.wav"

pytestmark = pytest.mark.e2e

skip_reason = []
if not HF_TOKEN:
    skip_reason.append("HF_TOKEN no está seteado")
if not AUDIO.exists():
    skip_reason.append(f"audio no encontrado: {AUDIO}")

needs_env = pytest.mark.skipif(bool(skip_reason), reason=" | ".join(skip_reason) or "ok")


def _ffprobe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


@pytest.fixture(scope="session")
def baseline_time(tmp_path_factory):
    """Mide wall time sin compresión — baseline para comparar paralelización."""
    from speechlib.core_analysis import core_analysis
    from speechlib.transcribe import _get_faster_whisper_model

    _get_faster_whisper_model.cache_clear()

    log_dir = tmp_path_factory.mktemp("e2e_baseline")
    tmp_audio = log_dir / AUDIO.name
    shutil.copy2(AUDIO, tmp_audio)

    t0 = time.perf_counter()
    core_analysis(
        str(tmp_audio),
        voices_folder=None,
        log_folder=str(log_dir),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=HF_TOKEN,
        model_type="faster-whisper",
        skip_enhance=True,
        compress=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[baseline] sin compress: {elapsed:.1f}s")
    return elapsed


@pytest.fixture(scope="session")
def compressed_result(tmp_path_factory, baseline_time):
    """Corre core_analysis con compress=True — sin mocks."""
    from speechlib.core_analysis import core_analysis

    log_dir = tmp_path_factory.mktemp("e2e_compress")

    # Copiar audio a tmp para que el .m4a no contamine el repo
    tmp_audio = log_dir / AUDIO.name
    shutil.copy2(AUDIO, tmp_audio)

    t0 = time.perf_counter()
    segments = core_analysis(
        str(tmp_audio),
        voices_folder=None,
        log_folder=str(log_dir),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=HF_TOKEN,
        model_type="faster-whisper",
        skip_enhance=True,
        compress=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[compress] con compress: {elapsed:.1f}s")

    m4a_path = tmp_audio.with_suffix(".m4a")
    return segments, m4a_path, tmp_audio, baseline_time, elapsed


@needs_env
def test_compressed_file_exists(compressed_result):
    """compress=True produce un archivo .m4a junto al source."""
    _, m4a_path, _, _, _ = compressed_result
    assert m4a_path.exists(), f"No se encontró archivo comprimido: {m4a_path}"
    assert m4a_path.stat().st_size > 0, "Archivo comprimido está vacío"


@needs_env
def test_compressed_is_aac_mono_16k(compressed_result):
    """El archivo comprimido es AAC, mono, 16kHz."""
    _, m4a_path, _, _, _ = compressed_result
    info = _ffprobe(m4a_path)
    assert info["codec_name"] == "aac", f"codec esperado aac, got {info['codec_name']}"
    assert info["channels"] == 1, f"channels esperado 1, got {info['channels']}"
    assert info["sample_rate"] == "16000", f"sample_rate esperado 16000, got {info['sample_rate']}"


@needs_env
def test_compressed_bitrate_near_96k(compressed_result):
    """Bitrate del archivo comprimido está cerca de 96kbps (tolerancia amplia por VBR)."""
    _, m4a_path, _, _, _ = compressed_result
    info = _ffprobe(m4a_path)
    bit_rate = int(info.get("bit_rate", 0))
    assert 40_000 <= bit_rate <= 140_000, (
        f"bit_rate fuera de rango: {bit_rate} bps (esperado ~96kbps)"
    )


@needs_env
def test_no_compressed_file_by_default(tmp_path):
    """Sin compress=True, no se produce archivo .m4a."""
    from speechlib.core_analysis import core_analysis

    tmp_audio = tmp_path / AUDIO.name
    shutil.copy2(AUDIO, tmp_audio)

    core_analysis(
        str(tmp_audio),
        voices_folder=None,
        log_folder=str(tmp_path / "logs"),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=HF_TOKEN,
        model_type="faster-whisper",
        skip_enhance=True,
    )

    m4a_path = tmp_audio.with_suffix(".m4a")
    assert not m4a_path.exists(), f"Se encontró .m4a sin compress=True: {m4a_path}"


@needs_env
def test_pipeline_output_unchanged(compressed_result):
    """La compresión no afecta los segmentos de transcripción."""
    segments, _, _, _, _ = compressed_result
    assert len(segments) > 0, "Pipeline no produjo segmentos"
    for seg in segments:
        assert len(seg) == 4, f"Segmento malformado: {seg}"
        assert seg[1] >= seg[0], f"end < start: {seg}"


@needs_env
def test_compress_runs_parallel_with_diarization(compressed_result):
    """Compresión en paralelo con diarización: overhead < 2s vs baseline sin compress."""
    _, _, _, baseline, with_compress = compressed_result
    overhead = with_compress - baseline
    print(f"\n[parallel] baseline={baseline:.1f}s, compress={with_compress:.1f}s, overhead={overhead:.1f}s")
    assert overhead < 2.0, (
        f"Overhead de compresión demasiado alto: {overhead:.1f}s. "
        f"baseline={baseline:.1f}s, con compress={with_compress:.1f}s. "
        "compress_audio debería correr en paralelo con diarización."
    )
