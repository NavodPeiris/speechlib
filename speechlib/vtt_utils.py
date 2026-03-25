"""
Utilidades de parsing y escritura para archivos WebVTT.

Funciones y tipos reutilizables por cualquier módulo de speechlib:
  VttBlock          — segmento parseado (dataclass)
  ts_to_ms()        — timestamp VTT → milisegundos
  seconds_to_vtt_ts() — segundos → timestamp VTT (HH:MM:SS.mmm)
  parse_vtt()       — leer VTT → (header, list[VttBlock])
  write_vtt()       — escribir VTT desde (header, list[VttBlock])
"""

import re
from dataclasses import dataclass
from pathlib import Path


TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
SPEAKER_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)", re.DOTALL)


@dataclass
class VttBlock:
    index: str
    start_ms: int
    end_ms: int
    speaker: str
    text: str
    raw_timestamp: str


def ts_to_ms(ts: str) -> int:
    """Convierte timestamp VTT 'HH:MM:SS.mmm' → milisegundos."""
    m = TS_RE.match(ts.strip())
    h, mn, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return ((h * 3600 + mn * 60 + s) * 1000) + ms


def seconds_to_vtt_ts(seconds: float) -> str:
    """Convierte segundos → timestamp VTT 'HH:MM:SS.mmm'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_vtt(path: Path) -> tuple[str, list[VttBlock]]:
    """Lee un archivo VTT y devuelve (header, list[VttBlock])."""
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
    """Escribe un archivo VTT a partir de (header, list[VttBlock])."""
    lines = [header, ""]
    for b in blocks:
        lines += [b.index, b.raw_timestamp, f"[{b.speaker}] {b.text}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
