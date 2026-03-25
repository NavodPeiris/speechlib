"""
Re-etiqueta los speakers de un VTT existente sin re-transcribir ni re-diarizar.

Extrae el audio chunk de cada segmento [unknown] desde el WAV ya procesado,
calcula el embedding y lo compara contra la libreria de voces.

Uso:
    python speechlib/tools/relabel_vtt.py VTT_PATH AUDIO_PATH VOICES_FOLDER [--threshold 0.40]

    VTT_PATH      : VTT a re-etiquetar
    AUDIO_PATH    : WAV procesado (mismo que genero el VTT, ej: _16k.wav)
    VOICES_FOLDER : carpeta con subdirectorios por speaker

Ejemplo:
    python speechlib/tools/relabel_vtt.py \\
        "C:\\workspace\\@recordings\\20260320 Patricio Renner\\Voz 260320_164522_16k_121520_es.vtt" \\
        "C:\\workspace\\@recordings\\20260320 Patricio Renner\\Voz 260320_164522_16k.wav" \\
        "C:\\workspace\\#dev\\speechlib\\transcript_samples\\voices"

Output: escribe VTT corregido junto al original con sufijo _relabeled.vtt
No modifica el archivo original.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")

import numpy as np
from speechlib.speaker_recognition import (
    get_embedding, cosine_similarity, find_best_speaker,
    load_avg_voice_embeddings, SPEAKER_SIMILARITY_THRESHOLD,
)
from speechlib.audio_utils import slice_and_save
from speechlib.vtt_utils import VttBlock, TS_RE, SPEAKER_RE, ts_to_ms, parse_vtt, write_vtt

DEFAULT_THRESHOLD = SPEAKER_SIMILARITY_THRESHOLD
DEFAULT_PAD_MIN_MS = 2000   # ventana minima para embedding cuando --pad-short activo


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtt_path")
    parser.add_argument("audio_path")
    parser.add_argument("voices_folder")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--pad-short",
        action="store_true",
        help="[EXPERIMENTAL] Segmentos mas cortos que --pad-min-ms se expanden "
             "simetricamente para el calculo del embedding. "
             "Los timestamps del VTT no cambian.",
    )
    parser.add_argument(
        "--pad-min-ms",
        type=int,
        default=DEFAULT_PAD_MIN_MS,
        help=f"Duracion minima en ms para embedding cuando --pad-short activo (default: {DEFAULT_PAD_MIN_MS})",
    )
    args = parser.parse_args()

    vtt_path = Path(args.vtt_path)
    audio_path = args.audio_path
    voices_folder = Path(args.voices_folder)
    threshold = args.threshold
    pad_short = args.pad_short
    pad_min_ms = args.pad_min_ms

    print(f"\nVTT      : {vtt_path.name}")
    print(f"Audio    : {Path(audio_path).name}")
    print(f"Voices   : {voices_folder}")
    print(f"Threshold: {threshold}")
    if pad_short:
        print(f"Padding  : EXPERIMENTAL — segmentos < {pad_min_ms}ms se expanden a {pad_min_ms}ms para embedding")

    print("\nCargando libreria de voces...")
    speaker_embs = load_avg_voice_embeddings(voices_folder)
    print(f"  {len(speaker_embs)} speakers: {sorted(speaker_embs)}")

    print("\nParsando VTT...")
    header, blocks = parse_vtt(vtt_path)
    unknown_count = sum(1 for b in blocks if b.speaker == "unknown")
    print(f"  {len(blocks)} segmentos totales, {unknown_count} [unknown]")

    tmp = Path(tempfile.mktemp(suffix=".wav"))
    changed = 0
    errors = 0

    print(f"\nRe-etiquetando {unknown_count} segmentos [unknown]...")
    for i, block in enumerate(blocks):
        if block.speaker != "unknown":
            continue

        try:
            duration_ms = block.end_ms - block.start_ms
            if pad_short and duration_ms < pad_min_ms:
                pad = (pad_min_ms - duration_ms) // 2
                extract_start = max(0, block.start_ms - pad)
                extract_end = block.end_ms + pad
            else:
                extract_start = block.start_ms
                extract_end = block.end_ms
            slice_and_save(audio_path, extract_start, extract_end, str(tmp))
            test_emb = get_embedding(str(tmp))
            new_speaker = find_best_speaker(test_emb, speaker_embs, threshold)
        except Exception as e:
            errors += 1
            continue

        if new_speaker != "unknown":
            block.speaker = new_speaker
            changed += 1

        if (i + 1) % 50 == 0 or i + 1 == len(blocks):
            pct = (i + 1) / len(blocks) * 100
            print(f"  [{pct:5.1f}%] {i+1}/{len(blocks)}  identificados: {changed}  errores: {errors}")

    tmp.unlink(missing_ok=True)

    suffix = "_relabeled_padded" if pad_short else "_relabeled"
    out_path = vtt_path.with_stem(vtt_path.stem + suffix)
    write_vtt(out_path, header, blocks)

    print(f"\n{'='*50}")
    print(f"  Segmentos re-etiquetados : {changed} / {unknown_count}")
    print(f"  Siguen como [unknown]    : {unknown_count - changed - errors}")
    print(f"  Errores                  : {errors}")
    print(f"  Output                   : {out_path.name}")
    print(f"{'='*50}")

    # Distribucion final de speakers
    from collections import Counter
    dist = Counter(b.speaker for b in blocks)
    print("\nDistribucion de speakers:")
    for speaker, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {speaker:<25} {count:>4} segmentos")


if __name__ == "__main__":
    main()
