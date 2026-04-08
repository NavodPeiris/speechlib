"""
Application service: re-etiquetar un transcript persistido contra una libreria
de voces actualizada.

Reemplaza la logica fragmentada de tools/relabel_vtt.py (que tenia dos ramas
contradictorias --rttm y --all-speakers, y donde el bug del literal "unknown"
vivia). Ahora todo el reconocimiento se reduce a:

    1. Cargar Transcript desde JSON
    2. Llamar assign_speakers (funcion pura del dominio)
    3. Guardar Transcript actualizado

assign_speakers preserva por construccion el SPEAKER_XX cuando ningun voice
supera threshold — el bug original es estructuralmente imposible aqui.

Mutable shell: solo I/O. Las decisiones (matching, threshold, fallback) viven
todas en el dominio.
"""

from pathlib import Path

import numpy as np

from ..domain.recognition import assign_speakers
from ..domain.transcript import Transcript


def relabel_transcript(
    src: Path,
    dst: Path,
    embeddings_by_tag: dict[str, np.ndarray],
    voice_library: dict[str, np.ndarray],
    threshold: float,
) -> Transcript:
    """Carga el transcript en src, re-evalua identidades y guarda en dst.

    Args:
        src: ruta al transcript.json existente.
        dst: ruta destino del transcript actualizado.
        embeddings_by_tag: {SPEAKER_XX: embedding} ya calculados desde el audio.
        voice_library: {nombre: embedding} de voces conocidas.
        threshold: minimo de cosine similarity para identificar.

    Returns:
        El Transcript actualizado (ya guardado en dst).
    """
    transcript = Transcript.load(src)
    relabeled = assign_speakers(
        transcript=transcript,
        embeddings_by_tag=embeddings_by_tag,
        voice_library=voice_library,
        threshold=threshold,
    )
    relabeled.save(dst)
    return relabeled
