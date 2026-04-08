"""
Adapter de migracion: convierte el formato legacy de core_analysis
([start_s, end_s, text, label] + speaker_map + annotation pyannote) en
un Transcript del nuevo dominio.

Permite que core_analysis siga produciendo su output legacy (VTT,
speaker_map.json) sin tocar la logica existente, mientras que en paralelo
publica el nuevo formato canonico (transcript.json) con SpeakerIdentity
completo, preservando el diarization_tag pyannote para los speakers no
identificados.

Esto cierra el "gap" del bug original: el dominio nuevo se construye con
informacion suficiente para que assign_speakers / relabel pueda re-evaluar
sin perder el SPEAKER_XX.

GOOS-sin-mocks: funcion pura. Recibe annotation_turns como lista de tuples
para no depender de pyannote en los tests.
"""

from typing import Iterable, Optional

from ..domain.transcript import SpeakerIdentity, Transcript, TranscriptSegment


def _find_best_overlap_tag(
    annotation_turns: list[tuple[float, float, str]],
    start_s: float,
    end_s: float,
) -> Optional[str]:
    """Devuelve el SPEAKER_XX cuyo turno tiene maximo overlap con [start_s, end_s].
    None si no hay ningun overlap > 0."""
    best_tag = None
    best_overlap = 0.0
    for t_start, t_end, tag in annotation_turns:
        overlap = min(t_end, end_s) - max(t_start, start_s)
        if overlap > best_overlap:
            best_overlap = overlap
            best_tag = tag
    return best_tag


def _build_identity(
    label: str,
    diarization_tag: Optional[str],
    speaker_map: dict[str, str],
) -> SpeakerIdentity:
    """Decide la SpeakerIdentity para un segmento legacy.

    Reglas:
    - Si diarization_tag esta disponible, se usa como tag pyannote.
    - Si no hay tag (no overlap encontrado), el label se usa como tag de fallback.
    - recognized_name se setea solo si label corresponde a un nombre real
      en speaker_map (es decir, label != tag y label != "unknown").
    - El literal "unknown" del legacy se NORMALIZA a recognized_name=None
      (defensa contra el bug original).
    """
    tag = diarization_tag or label

    # Caso: el label es identico al tag (no identificado en legacy)
    if label == tag or label.startswith("SPEAKER_") or label == "unknown":
        return SpeakerIdentity(diarization_tag=tag, recognized_name=None)

    # Caso: el label es un nombre real
    return SpeakerIdentity(
        diarization_tag=tag,
        recognized_name=label,
        similarity=None,  # legacy no expone score
    )


def build_transcript_from_legacy_segments(
    legacy_segments: Iterable,
    annotation_turns: list[tuple[float, float, str]],
    speaker_map: dict[str, str],
    audio_path: str,
    language: str,
) -> Transcript:
    """Convierte el output de core_analysis (formato legacy) a Transcript.

    Args:
        legacy_segments: iterable de [start_s, end_s, text, label] (lista o tuple).
        annotation_turns: turnos pyannote como (start_s, end_s, SPEAKER_XX).
            Se usa para resolver el diarization_tag de cada segmento por overlap.
        speaker_map: mapa tag -> name del legacy. No se usa para la resolucion
            (esa es por overlap), pero se acepta por completitud.
        audio_path, language: metadata para el aggregate.

    Returns:
        Transcript inmutable con SpeakerIdentity completo en cada segmento.
    """
    new_segments = []
    for legacy in legacy_segments:
        start_s, end_s, text, label = legacy[0], legacy[1], legacy[2], legacy[3]
        # Para no identificados, el legacy label YA es el SPEAKER_XX correcto:
        # core_analysis hace speaker_map[tag]=tag cuando el reconocimiento falla.
        # Tras absorb_micro_segments + merge_short_turns un segmento puede cubrir
        # multiples turnos pyannote, asi que el overlap puede devolver el tag
        # mas grande en lugar del verdadero. Saltamos el overlap y confiamos
        # en la fuente de verdad legacy.
        if label.startswith("SPEAKER_"):
            diarization_tag = label
        else:
            diarization_tag = _find_best_overlap_tag(annotation_turns, start_s, end_s)
        identity = _build_identity(label, diarization_tag, speaker_map)
        new_segments.append(
            TranscriptSegment(
                start_ms=int(round(start_s * 1000)),
                end_ms=int(round(end_s * 1000)),
                text=text,
                speaker=identity,
            )
        )
    return Transcript(
        segments=tuple(new_segments),
        audio_path=audio_path,
        language=language,
    )
