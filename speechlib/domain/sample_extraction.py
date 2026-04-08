"""
Domain logic para extraer audio samples por speaker.

plan_speaker_samples es una funcion pura que decide QUE recortar de un audio:
agrupa los segmentos del Transcript por SpeakerIdentity.label (de modo que
los identificados colapsen aunque vengan de SPEAKER_XX distintos, y los
no identificados queden separados por su tag pyannote), filtra los que no
llegan a la duracion minima y selecciona los top-N mas largos por speaker.

Reemplaza la logica de extract_unknown_speakers porque cubre TANTO speakers
identificados como no identificados en una sola operacion.

Estilo GOOS-sin-mocks: cero I/O. La ejecucion (cortar el audio y escribir
WAVs) vive en speechlib/services/extract_samples.py.
"""

from dataclasses import dataclass
from typing import Iterable

from .transcript import Transcript, TranscriptSegment


@dataclass(frozen=True)
class SampleClip:
    """Una ventana temporal a recortar del audio fuente."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class SpeakerSamplePlan:
    """Plan de extraccion para un speaker: que clips cortar."""

    speaker_label: str
    is_identified: bool
    clips: tuple[SampleClip, ...]


def plan_speaker_samples(
    transcript: Transcript,
    max_clips_per_speaker: int,
    min_clip_duration_ms: int,
) -> tuple[SpeakerSamplePlan, ...]:
    """Construye los planes de extraccion para todos los speakers del transcript.

    Args:
        transcript: aggregate del que se extraen los segmentos.
        max_clips_per_speaker: cap del top-N por speaker. Si <=0, retorna ().
        min_clip_duration_ms: duracion minima por clip; los que no llegan
            quedan filtrados.

    Returns:
        Tuple de planes ordenado: identificados primero (alfabetico) y luego
        no identificados (alfabetico). Speakers cuyos clips quedan todos
        filtrados no aparecen en el resultado.
    """
    if max_clips_per_speaker <= 0:
        return ()

    # Agrupar segmentos por label (no por tag): identificados con el mismo
    # nombre colapsan, no identificados quedan separados por SPEAKER_XX.
    grouped: dict[str, list[TranscriptSegment]] = {}
    identified_label: dict[str, bool] = {}
    for seg in transcript.segments:
        label = seg.speaker.label
        grouped.setdefault(label, []).append(seg)
        # is_identified es invariante por label: si algun segmento del label
        # tiene recognized_name, todos lo tienen (label = recognized_name).
        identified_label[label] = seg.speaker.is_identified

    plans: list[SpeakerSamplePlan] = []
    for label, segments in grouped.items():
        # Filtrar por duracion minima
        eligible = [s for s in segments if (s.end_ms - s.start_ms) >= min_clip_duration_ms]
        if not eligible:
            continue
        # Top-N por duracion descendente
        top = sorted(eligible, key=lambda s: -(s.end_ms - s.start_ms))[:max_clips_per_speaker]
        # Reordenar por start_ms para output consistente
        top.sort(key=lambda s: s.start_ms)
        clips = tuple(SampleClip(start_ms=s.start_ms, end_ms=s.end_ms) for s in top)
        plans.append(
            SpeakerSamplePlan(
                speaker_label=label,
                is_identified=identified_label[label],
                clips=clips,
            )
        )

    # Identificados primero (alfabetico), luego no identificados (alfabetico).
    plans.sort(key=lambda p: (not p.is_identified, p.speaker_label))
    return tuple(plans)
