"""AT: el pipeline no debe producir micro-segmentos en el resultado final.

Brecha actual: la diarización produce micro-segmentos < 0.3s cuando pyannote
corta speaker boundaries en medio de frases (ej: "la" sola en 0.2s,
"De" en 0.1s). Estos fragmentos degradan la calidad de la transcripción.

Los overlaps genuinos (dos speakers hablando simultáneamente) deben preservarse.

Corre core_analysis E2E con audio real + pyannote real + whisper real.
"""
import os
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

needs_hf = pytest.mark.skipif(bool(skip_reason), reason=" | ".join(skip_reason) or "ok")


@pytest.fixture(scope="session")
def run_result(tmp_path_factory):
    """Corre core_analysis una vez con audio real — sin mocks."""
    from speechlib.core_analysis import core_analysis
    from speechlib.transcribe import _get_faster_whisper_model

    _get_faster_whisper_model.cache_clear()

    log_dir = tmp_path_factory.mktemp("e2e_microseg")

    result = core_analysis(
        str(AUDIO),
        voices_folder=None,
        log_folder=str(log_dir),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=HF_TOKEN,
        model_type="faster-whisper",
    )

    return result


@needs_hf
def test_no_micro_segments_under_threshold(run_result):
    """Ningún segmento del resultado final debe durar < 0.3s.

    Brecha: la diarización produce micro-segmentos de 0.0-0.2s cuando pyannote
    corta speaker boundaries en medio de frases. Estos fragmentos no aportan
    valor y rompen la lectura de la transcripción.
    """
    THRESHOLD = 0.3

    micros = []
    for seg in run_result:
        duration = seg[1] - seg[0]
        if duration < THRESHOLD:
            micros.append(f"  [{seg[0]}-{seg[1]}] = {duration:.2f}s  {seg[3]}: {seg[2][:50]}")

    assert len(micros) == 0, (
        f"{len(micros)} micro-segmentos < {THRESHOLD}s detectados:\n" + "\n".join(micros[:10])
    )
