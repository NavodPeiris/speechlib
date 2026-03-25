"""
Revierte etiquetas de speaker en un VTT reemplazándolas por unknown_001, 002, ...

Los speakers a revertir se mapean en orden de primera aparición en el VTT,
independientemente del orden en que se pasen como argumento.

Uso:
    python speechlib/tools/revert_speakers.py VTT_PATH SPEAKER1 [SPEAKER2 ...]

Ejemplo:
    python speechlib/tools/revert_speakers.py \\
        "transcript_es.vtt" \\
        "Jolyon" "Patricio Renner" "Francisco" "Ina Gonzalez"

Output: {stem}_reverted.vtt en el mismo directorio. El original no se modifica.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speechlib.tools.relabel_vtt import parse_vtt, write_vtt


def revert_speakers(vtt_path: Path, speakers_to_revert: list[str]) -> Path:
    """
    Reemplaza los speakers indicados con unknown_NNN (orden de primera aparición).

    Returns: Path al archivo de salida {stem}_reverted.vtt.
    """
    speakers_set = set(speakers_to_revert)
    header, blocks = parse_vtt(vtt_path)

    # Determinar orden de primera aparición en el VTT
    first_appearance_order: list[str] = []
    for block in blocks:
        if block.speaker in speakers_set and block.speaker not in first_appearance_order:
            first_appearance_order.append(block.speaker)

    # Construir mapeo speaker → unknown_NNN
    mapping = {
        speaker: f"unknown_{i+1:03d}"
        for i, speaker in enumerate(first_appearance_order)
    }

    # Aplicar mapeo
    for block in blocks:
        if block.speaker in mapping:
            block.speaker = mapping[block.speaker]

    out_path = vtt_path.with_stem(vtt_path.stem + "_reverted")
    write_vtt(out_path, header, blocks)

    return out_path


def main():
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtt_path", help="VTT a corregir")
    parser.add_argument("speakers", nargs="+", help="Speakers a revertir a unknown_NNN")
    args = parser.parse_args()

    vtt_path = Path(args.vtt_path)
    out = revert_speakers(vtt_path, args.speakers)

    _, blocks = parse_vtt(out)
    dist = Counter(b.speaker for b in blocks)

    print(f"\nOutput: {out}")
    print(f"\nDistribucion de speakers:")
    for speaker, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {speaker:<30} {count:>4} segmentos")


if __name__ == "__main__":
    main()
