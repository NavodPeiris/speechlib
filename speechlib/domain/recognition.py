"""
Funcion pura de reconocimiento de speakers contra una libreria de voces.

assign_speakers es la operacion central del refactor: reemplaza la logica
duplicada que vivia repartida entre core_analysis (loop de speaker_recognition
+ fallback "unknown" → tag) y relabel_vtt (dos ramas: --rttm y --all-speakers).

Estilo GOOS-sin-mocks: la funcion no toca audio, ni filesystem, ni pyannote.
Recibe embeddings ya calculados como dict[tag → ndarray] y retorna un nuevo
Transcript. Toda la I/O se queda en la capa de application services.

Invariante critico: cuando ningun voice supera threshold, el segmento
conserva su SpeakerIdentity con recognized_name=None — el diarization_tag
NUNCA se pierde y label() siempre cae al SPEAKER_XX original. El bug del
relabel --all-speakers es estructuralmente imposible aqui.
"""

from typing import Optional

import numpy as np

from .transcript import SpeakerIdentity, Transcript


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _best_match(
    embedding: np.ndarray,
    voice_library: dict[str, np.ndarray],
    threshold: float,
) -> tuple[Optional[str], Optional[float]]:
    """Devuelve (name, similarity) del mejor match, o (None, best_similarity)
    si ningun voice supera el threshold. None es la senal de 'no identificado',
    NO el string 'unknown'."""
    best_name: Optional[str] = None
    best_score = -1.0
    for name, voice_emb in voice_library.items():
        score = _cosine_similarity(embedding, voice_emb)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is None:
        return None, None  # libreria vacia
    if best_score < threshold:
        return None, best_score
    return best_name, best_score


def assign_speakers(
    transcript: Transcript,
    embeddings_by_tag: dict[str, np.ndarray],
    voice_library: dict[str, np.ndarray],
    threshold: float,
) -> Transcript:
    """Asigna identidades de speaker a cada segmento del transcript.

    Para cada diarization_tag presente en el transcript:
    - Si tiene embedding en embeddings_by_tag, se compara contra voice_library.
      El resultado (con o sin match) se aplica a TODOS los segmentos del tag.
    - Si NO tiene embedding, los segmentos de ese tag quedan intactos.

    Retorna un Transcript nuevo. La entrada no se mutua.
    """
    # Resolver identidad UNA vez por tag (todos los segmentos del mismo tag
    # comparten identidad). Las identidades sin embedding se marcan como None
    # para indicar "no tocar".
    new_identity_by_tag: dict[str, Optional[SpeakerIdentity]] = {}
    for tag in transcript.diarization_tags:
        embedding = embeddings_by_tag.get(tag)
        if embedding is None:
            new_identity_by_tag[tag] = None
            continue
        name, similarity = _best_match(embedding, voice_library, threshold)
        new_identity_by_tag[tag] = SpeakerIdentity(
            diarization_tag=tag,
            recognized_name=name,
            similarity=similarity,
        )

    new_segments = tuple(
        seg if new_identity_by_tag[seg.speaker.diarization_tag] is None
        else seg.with_speaker(new_identity_by_tag[seg.speaker.diarization_tag])
        for seg in transcript.segments
    )
    return transcript.with_segments(new_segments)
