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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")

import numpy as np
from speechlib.speaker_recognition import get_embedding, cosine_similarity
from speechlib.audio_utils import slice_and_save

DEFAULT_THRESHOLD = 0.40
VOICES_SKIP_PREFIX = "_"

TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
SPEAKER_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)", re.DOTALL)


# ── VTT parsing ───────────────────────────────────────────────────────────────

@dataclass
class VttBlock:
    index: str
    start_ms: int
    end_ms: int
    speaker: str
    text: str
    raw_timestamp: str


def ts_to_ms(ts: str) -> int:
    m = TS_RE.match(ts.strip())
    h, mn, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return ((h * 3600 + mn * 60 + s) * 1000) + ms


def parse_vtt(path: Path) -> tuple[str, list[VttBlock]]:
    """Returns (header_line, blocks)."""
    text = path.read_text(encoding="utf-8")
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    header = raw_blocks[0]  # "WEBVTT"
    blocks: list[VttBlock] = []

    for raw in raw_blocks[1:]:
        lines = raw.splitlines()
        if len(lines) < 3:
            continue

        index = lines[0]
        ts_line = lines[1]
        content = " ".join(lines[2:])

        if "-->" not in ts_line:
            continue

        start_str, end_str = ts_line.split("-->")
        start_ms = ts_to_ms(start_str)
        end_ms = ts_to_ms(end_str)

        m = SPEAKER_RE.match(content)
        if m:
            speaker, text = m[1], m[2].strip()
        else:
            speaker, text = "unknown", content.strip()

        blocks.append(VttBlock(
            index=index,
            start_ms=start_ms,
            end_ms=end_ms,
            speaker=speaker,
            text=text,
            raw_timestamp=ts_line.strip(),
        ))

    return header, blocks


def write_vtt(path: Path, header: str, blocks: list[VttBlock]) -> None:
    lines = [header, ""]
    for b in blocks:
        lines += [b.index, b.raw_timestamp, f"[{b.speaker}] {b.text}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Voice library ─────────────────────────────────────────────────────────────

def load_avg_embeddings(voices_folder: Path) -> dict[str, np.ndarray]:
    """Returns {speaker: avg_embedding}."""
    result = {}
    for entry in sorted(voices_folder.iterdir()):
        if not entry.is_dir() or entry.name.startswith(VOICES_SKIP_PREFIX):
            continue
        embs = []
        for wav in sorted(entry.glob("*.wav")):
            try:
                embs.append(get_embedding(str(wav)))
            except Exception:
                pass
        if embs:
            result[entry.name] = np.mean(embs, axis=0)
    return result


def identify(test_emb: np.ndarray, speaker_embs: dict[str, np.ndarray], threshold: float) -> str:
    best, best_score = "unknown", -1.0
    for speaker, emb in speaker_embs.items():
        score = cosine_similarity(test_emb, emb)
        if score > best_score:
            best_score = score
            best = speaker
    return best if best_score >= threshold else "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtt_path")
    parser.add_argument("audio_path")
    parser.add_argument("voices_folder")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    vtt_path = Path(args.vtt_path)
    audio_path = args.audio_path
    voices_folder = Path(args.voices_folder)
    threshold = args.threshold

    print(f"\nVTT      : {vtt_path.name}")
    print(f"Audio    : {Path(audio_path).name}")
    print(f"Voices   : {voices_folder}")
    print(f"Threshold: {threshold}")

    print("\nCargando libreria de voces...")
    speaker_embs = load_avg_embeddings(voices_folder)
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
            slice_and_save(audio_path, block.start_ms, block.end_ms, str(tmp))
            test_emb = get_embedding(str(tmp))
            new_speaker = identify(test_emb, speaker_embs, threshold)
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

    out_path = vtt_path.with_stem(vtt_path.stem + "_relabeled")
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
