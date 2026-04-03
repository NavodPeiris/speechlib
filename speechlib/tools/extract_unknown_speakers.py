"""
Extrae segmentos de audio de speakers desconocidos a disco para que el usuario
los nombre e incorpore a la librería de voces.

Uso típico: después de diarización + speaker_recognition, los speakers cuyo
score quedó bajo el threshold (retornaron "unknown") se pasan aquí para
guardar sus segmentos más representativos.

Estructura de salida:
    output_dir/
        SPEAKER_01_meeting/
            segment_01.wav   ← segmento más largo
            segment_02.wav
        SPEAKER_02_meeting/
            segment_01.wav
"""

import logging
from pathlib import Path

from ..audio_utils import slice_and_save

logger = logging.getLogger(__name__)


def extract_unknown_speakers(
    audio_path: Path,
    unknown_segments: dict[str, list[list[float]]],
    output_dir: Path,
    min_duration_s: float = 2.0,
    max_clips: int = 4,
) -> dict[str, Path]:
    """Guarda clips representativos de speakers desconocidos.

    Args:
        audio_path: WAV procesado (normalizado/enhanced).
        unknown_segments: {speaker_tag: [[start_s, end_s], ...]}
        output_dir: Directorio raíz para _unknown speakers.
        min_duration_s: Duración mínima de un clip para ser guardado.
        max_clips: Máximo de clips por speaker (se priorizan los más largos).

    Returns:
        {speaker_tag: Path} con la carpeta creada por cada speaker.
    """
    audio_stem = Path(audio_path).stem
    result: dict[str, Path] = {}

    for speaker_tag, segments in unknown_segments.items():
        # Filtrar por duración mínima y ordenar de mayor a menor
        valid = sorted(
            [s for s in segments if (s[1] - s[0]) >= min_duration_s],
            key=lambda s: s[1] - s[0],
            reverse=True,
        )

        if not valid:
            logger.info("Speaker %s: sin segmentos >= %.1fs, omitido", speaker_tag, min_duration_s)
            continue

        selected = valid[:max_clips]
        speaker_dir = output_dir / f"{speaker_tag}_{audio_stem}"
        speaker_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(selected, start=1):
            dest = str(speaker_dir / f"segment_{i:02d}.wav")
            slice_and_save(str(audio_path), seg[0] * 1000, seg[1] * 1000, dest)

        logger.info(
            "Speaker %s: %d clips guardados en %s", speaker_tag, len(selected), speaker_dir
        )
        result[speaker_tag] = speaker_dir

    return result
