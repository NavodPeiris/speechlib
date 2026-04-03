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

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")

import numpy as np
from speechlib.speaker_recognition import (
    get_embedding,
    cosine_similarity,
    find_best_speaker,
    load_avg_voice_embeddings,
    SPEAKER_SIMILARITY_THRESHOLD,
    is_unidentified_speaker,
)
from speechlib.audio_utils import slice_and_save
from speechlib.vtt_utils import (
    VttBlock,
    TS_RE,
    SPEAKER_RE,
    ts_to_ms,
    parse_vtt,
    write_vtt,
)

DEFAULT_THRESHOLD = SPEAKER_SIMILARITY_THRESHOLD
DEFAULT_PAD_MIN_MS = 2000  # ventana minima para embedding cuando --pad-short activo
MIN_OVERLAP_S = 0.3  # minimum overlap for block-to-speaker matching


def match_block_to_speaker(
    block_start_s: float, block_end_s: float, annotation
) -> str | None:
    """Match a VTT block to a speaker tag by maximum overlap."""
    best_spk, best_overlap = None, 0.0
    for turn, _, spk in annotation.itertracks(yield_label=True):
        overlap = min(turn.end, block_end_s) - max(turn.start, block_start_s)
        if overlap > best_overlap:
            best_overlap, best_spk = overlap, spk
    if best_overlap < MIN_OVERLAP_S:
        return None
    return best_spk


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
    parser.add_argument(
        "--all-speakers",
        action="store_true",
        help="Re-evaluar TODOS los bloques del VTT (no solo [unknown]). "
        "Permite detectar misidentificaciones: bloques ya nombrados cuyo "
        "audio no supera threshold se reetiquetan a [unknown].",
    )
    parser.add_argument(
        "--rttm",
        type=Path,
        help="Path to diarization.rttm file. If provided, groups VTT blocks by SPEAKER_XX "
        "and runs speaker_recognition once per group instead of once per block.",
    )
    parser.add_argument(
        "--speaker-map",
        type=Path,
        help="Path to speaker_map.json. If provided with --rttm, applies mapping directly "
        "without computing embeddings.",
    )
    args = parser.parse_args()

    vtt_path = Path(args.vtt_path)
    audio_path = args.audio_path
    voices_folder = Path(args.voices_folder)
    threshold = args.threshold
    pad_short = args.pad_short
    pad_min_ms = args.pad_min_ms
    all_speakers = args.all_speakers
    rttm_path = args.rttm
    speaker_map_path = args.speaker_map

    print(f"\nVTT      : {vtt_path.name}")
    print(f"Audio    : {Path(audio_path).name}")
    print(f"Voices   : {voices_folder}")
    print(f"Threshold: {threshold}")
    if all_speakers:
        print("Modo     : --all-speakers (re-evalua todos los bloques)")
    if pad_short:
        print(
            f"Padding  : EXPERIMENTAL — segmentos < {pad_min_ms}ms se expanden a {pad_min_ms}ms para embedding"
        )

    print("\nParsando VTT...")
    header, blocks = parse_vtt(vtt_path)
    unknown_count = sum(1 for b in blocks if is_unidentified_speaker(b.speaker))
    target_count = len(blocks) if all_speakers else unknown_count
    print(f"  {len(blocks)} segmentos totales, {unknown_count} no identificados")

    annotation = None
    speaker_map = None
    use_rttm_grouping = rttm_path is not None
    use_speaker_map_only = rttm_path is not None and speaker_map_path is not None

    if rttm_path and rttm_path.exists():
        from pyannote.database.util import load_rttm

        annotation = next(iter(load_rttm(str(rttm_path)).values()))
        print(f"  RTTM loaded: {rttm_path.name}")

    if speaker_map_path and speaker_map_path.exists():
        import json

        speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
        print(f"  speaker_map loaded: {speaker_map_path.name}")

    if use_speaker_map_only:
        print("\nModo: --rttm + --speaker-map (aplicando mapeo directo sin embeddings)")
        for block in blocks:
            spk_tag = match_block_to_speaker(
                block.start_ms / 1000, block.end_ms / 1000, annotation
            )
            if spk_tag and spk_tag in speaker_map:
                block.speaker = speaker_map[spk_tag]
        changed = sum(1 for b in blocks if not is_unidentified_speaker(b.speaker))
        errors = 0
    elif use_rttm_grouping:
        print("\nModo: --rttm (agrupando por SPEAKER_XX)")
        from speechlib.speaker_recognition import speaker_recognition

        print("Cargando libreria de voces...")
        speaker_embs = load_avg_voice_embeddings(voices_folder)
        print(f"  {len(speaker_embs)} speakers: {sorted(speaker_embs)}")

        block_groups = {}
        for block in blocks:
            spk_tag = match_block_to_speaker(
                block.start_ms / 1000, block.end_ms / 1000, annotation
            )
            if spk_tag:
                if spk_tag not in block_groups:
                    block_groups[spk_tag] = []
                block_groups[spk_tag].append(block)

        for spk_tag, group_blocks in block_groups.items():
            segments = [
                [b.start_ms / 1000, b.end_ms / 1000, spk_tag] for b in group_blocks
            ]
            spk_name = speaker_recognition(
                str(audio_path), str(voices_folder), segments
            )
            for block in group_blocks:
                block.speaker = spk_name if spk_name != "unknown" else spk_tag

        changed = sum(1 for b in blocks if not is_unidentified_speaker(b.speaker))
        errors = 0
    else:
        print("\nCargando libreria de voces...")
        speaker_embs = load_avg_voice_embeddings(voices_folder)
        print(f"  {len(speaker_embs)} speakers: {sorted(speaker_embs)}")

        tmp = Path(tempfile.mktemp(suffix=".wav"))
        changed = 0
        errors = 0

        label = "todos los" if all_speakers else f"{unknown_count} [unknown]"
        print(f"\nRe-etiquetando {label} segmentos...")
        for i, block in enumerate(blocks):
            if not all_speakers and not is_unidentified_speaker(block.speaker):
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

            if new_speaker != block.speaker and (
                new_speaker != "unknown" or all_speakers
            ):
                block.speaker = new_speaker
                changed += 1

            if (i + 1) % 50 == 0 or i + 1 == len(blocks):
                pct = (i + 1) / len(blocks) * 100
                print(
                    f"  [{pct:5.1f}%] {i + 1}/{len(blocks)}  identificados: {changed}  errores: {errors}"
                )

        tmp.unlink(missing_ok=True)

    suffix = "_relabeled_padded" if pad_short else "_relabeled"
    out_path = vtt_path.with_stem(vtt_path.stem + suffix)
    write_vtt(out_path, header, blocks)

    print(f"\n{'=' * 50}")
    print(f"  Segmentos re-etiquetados : {changed} / {target_count}")
    print(
        f"  Siguen sin identificar  : {sum(1 for b in blocks if is_unidentified_speaker(b.speaker))}"
    )
    print(f"  Errores                  : {errors}")
    print(f"  Output                   : {out_path.name}")
    print(f"{'=' * 50}")

    # Distribucion final de speakers
    from collections import Counter

    dist = Counter(b.speaker for b in blocks)
    print("\nDistribucion de speakers:")
    for speaker, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {speaker:<25} {count:>4} segmentos")


if __name__ == "__main__":
    main()
