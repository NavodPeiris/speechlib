"""
Diagnostica por qué un tramo de audio es etiquetado como [unknown].

Uso:
    python speechlib/tools/diagnose_speaker.py AUDIO TIMESTAMP VOICES_FOLDER

    AUDIO          : path al archivo de audio (cualquier formato soportado por ffmpeg)
    TIMESTAMP      : HH:MM:SS del tramo a diagnosticar
    VOICES_FOLDER  : carpeta con subdirectorios por speaker (cada uno con WAVs)

Ejemplo:
    python speechlib/tools/diagnose_speaker.py \\
        "C:\\workspace\\@recordings\\20260320 Patricio Renner\\Voz 260320_164522.m4a" \\
        01:42:17 \\
        "C:\\workspace\\#dev\\speechlib\\transcript_samples\\voices"

Output: tabla de cosine similarity por speaker vs threshold actual (0.40).
No modifica nada.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")

import numpy as np
from speechlib.speaker_recognition import get_embedding, cosine_similarity, SPEAKER_SIMILARITY_THRESHOLD
from speechlib.audio_utils import extract_audio_segment

THRESHOLD = SPEAKER_SIMILARITY_THRESHOLD
WINDOW_S  = 30          # segundos a extraer alrededor del timestamp
VOICES_SKIP_PREFIX = "_"


def hms_to_seconds(ts: str) -> int:
    """'HH:MM:SS' → segundos enteros."""
    parts = ts.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def extract_window(audio_path: str, start_s: int, duration_s: int) -> Path:
    tmp_dir = Path(audio_path).parent / "tmp"
    tmp = tmp_dir / f"diag_{start_s}.wav"
    return extract_audio_segment(audio_path, tmp, start_s=max(0, start_s), duration_s=duration_s)


def load_speaker_embeddings(voices_folder: Path) -> dict[str, list[np.ndarray]]:
    """Retorna {speaker_name: [emb_per_segment, ...]}."""
    result = {}
    for entry in sorted(voices_folder.iterdir()):
        if not entry.is_dir() or entry.name.startswith(VOICES_SKIP_PREFIX):
            continue
        embs = []
        for wav in sorted(entry.glob("*.wav")):
            try:
                embs.append(get_embedding(str(wav)))
            except Exception as e:
                print(f"  [WARN] {entry.name}/{wav.name}: {e}")
        if embs:
            result[entry.name] = embs
    return result


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    audio_path   = sys.argv[1]
    timestamp    = sys.argv[2]
    voices_folder = Path(sys.argv[3])

    start_s = hms_to_seconds(timestamp)

    print(f"\nAudio    : {audio_path}")
    print(f"Timestamp: {timestamp}  ({start_s}s)")
    print(f"Ventana  : {start_s}s -> {start_s + WINDOW_S}s  ({WINDOW_S}s)")
    print(f"Threshold: {THRESHOLD}")

    print("\nExtrayendo ventana de audio...")
    tmp_wav = extract_window(audio_path, start_s, WINDOW_S)
    print(f"  -> {tmp_wav}  ({tmp_wav.stat().st_size // 1024} KB)")

    print("\nCalculando embedding del tramo...")
    try:
        test_emb = get_embedding(str(tmp_wav))
    except Exception as e:
        print(f"  [ERROR] No se pudo calcular embedding: {e}")
        tmp_wav.unlink(missing_ok=True)
        sys.exit(1)

    print("\nCargando librería de voces...")
    speaker_embs = load_speaker_embeddings(voices_folder)
    print(f"  {len(speaker_embs)} speakers: {sorted(speaker_embs)}")

    print("\n" + "=" * 62)
    print(f"  {'SPEAKER':<20} {'MIN':>6} {'AVG':>6} {'MAX':>6}  RESULT")
    print("=" * 62)

    rows = []
    for speaker, embs in speaker_embs.items():
        scores = [cosine_similarity(test_emb, e) for e in embs]
        rows.append((speaker, min(scores), np.mean(scores), max(scores)))

    rows.sort(key=lambda r: r[2], reverse=True)

    for speaker, s_min, s_avg, s_max in rows:
        result = "PASS" if s_avg >= THRESHOLD else "FAIL  (unknown)"
        print(f"  {speaker:<20} {s_min:>6.3f} {s_avg:>6.3f} {s_max:>6.3f}  {result}")

    print("=" * 62)

    best_speaker, _, best_avg, _ = rows[0]
    if best_avg < THRESHOLD:
        print(f"\n[DIAGNOSIS] Ningún speaker supera threshold={THRESHOLD}.")
        print(f"  Mejor candidato: {best_speaker} (avg={best_avg:.3f})")
        print(f"  Brecha vs threshold: {THRESHOLD - best_avg:+.3f}")
        print("  => H1 probable: cosine similarity insuficiente en este tramo.")
    else:
        print(f"\n[DIAGNOSIS] {best_speaker} supera threshold (avg={best_avg:.3f}).")
        print("  Si el VTT dice [unknown] aquí, el VTT fue generado con un run anterior.")

    tmp_wav.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
