"""
Application service que ejecuta SpeakerSamplePlan contra un audio real.

Mutable shell del dominio puro plan_speaker_samples: orquesta el I/O
(slicing del audio + escritura de WAVs) sin razonar sobre que extraer.

Estructura de salida:
    <output_dir>/<speaker_label>/clip_NN.wav

NOTA: speaker_label se usa tal cual como nombre de directorio. Espacios
y guiones se preservan; el usuario decidio no sanitizar (Windows + Unix
manejan ambos).
"""

from pathlib import Path

from ..audio_utils import slice_and_save
from ..domain.sample_extraction import SpeakerSamplePlan


def extract_speaker_samples(
    plans: tuple[SpeakerSamplePlan, ...],
    audio_path: Path,
    output_dir: Path,
) -> dict[str, list[Path]]:
    """Ejecuta los planes contra el audio fuente.

    Args:
        plans: tuple de SpeakerSamplePlan generados por plan_speaker_samples.
        audio_path: WAV fuente (usualmente el preprocessed/enhanced).
        output_dir: raiz donde se crearan los subdirectorios por speaker.

    Returns:
        {speaker_label: [paths_de_wavs_creados]} en el mismo orden que el plan.
    """
    output_dir = Path(output_dir)
    audio_path = Path(audio_path)
    written: dict[str, list[Path]] = {}

    for plan in plans:
        speaker_dir = output_dir / plan.speaker_label
        speaker_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, clip in enumerate(plan.clips, start=1):
            dest = speaker_dir / f"clip_{i:02d}.wav"
            slice_and_save(
                str(audio_path),
                clip.start_ms,
                clip.end_ms,
                str(dest),
            )
            paths.append(dest)
        written[plan.speaker_label] = paths

    return written
