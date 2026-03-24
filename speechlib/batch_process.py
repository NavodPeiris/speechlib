"""
Procesa múltiples carpetas de audio, identifica speakers conocidos,
extrae desconocidos a disco para que el usuario los nombre.

Uso:
    from speechlib.batch_process import batch_process

    report = batch_process(
        folders=[Path("@recordings/20260320 Patricio Renner"),
                 Path("@recordings/20260318 Ina TRE")],
        voices_folder=Path("transcript_samples/voices"),
        language="es",
        access_token=os.environ["HF_TOKEN"],
    )
    report.print_summary()
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .core_analysis import core_analysis
from .extract_unknown_speakers import extract_unknown_speakers

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".opus"}


@dataclass
class BatchReport:
    folders: list[Path] = field(default_factory=list)
    processed_files: list[Path] = field(default_factory=list)
    identified_speakers: set[str] = field(default_factory=set)
    unknown_speakers: list[dict] = field(default_factory=list)
    errors: int = 0

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  BATCH PROCESS REPORT")
        print("=" * 60)
        print(f"  Carpetas procesadas : {len(self.folders)}")
        print(f"  Archivos procesados : {len(self.processed_files)}")
        print(f"  Errores             : {self.errors}")
        print(f"  Speakers conocidos  : {sorted(self.identified_speakers)}")
        print(f"  Speakers desconocidos: {len(self.unknown_speakers)}")
        for u in self.unknown_speakers:
            print(f"    [{u['tag']}] en {u['audio'].name}  →  {u['folder']}")
        print("=" * 60)
        if self.unknown_speakers:
            print("\n  PRÓXIMOS PASOS:")
            print("  1. Escucha los clips en voices/_unknown/")
            print("  2. Renombra la carpeta con el nombre real de la persona")
            print("     ej: mv _unknown/SPEAKER_01_recording voices/Patricio")
            print("  3. Re-ejecuta el batch — esa persona será identificada")
        print()


def batch_process(
    folders: list[Path],
    voices_folder: Path,
    language: str,
    access_token: str,
    model_size: str = "large-v3-turbo",
    unknown_output_dir: Path | None = None,
    min_unknown_duration_s: float = 2.0,
    max_unknown_clips: int = 4,
    skip_enhance: bool = False,
) -> BatchReport:
    """Procesa múltiples carpetas de audio.

    Args:
        folders: Lista de carpetas a procesar (cada una puede tener varios audios).
        voices_folder: Librería de voces conocidas.
        language: Código ISO del idioma (es, en, ...).
        access_token: HuggingFace token para pyannote.
        model_size: Tamaño del modelo Whisper.
        unknown_output_dir: Dónde guardar clips de speakers desconocidos.
                            Por defecto: voices/_unknown/ relativo a voices_folder.
        min_unknown_duration_s: Duración mínima de clip para guardar.
        max_unknown_clips: Máximo de clips por speaker desconocido.
        skip_enhance: Omitir enhance_audio (más rápido, menor calidad).

    Returns:
        BatchReport con resumen completo.
    """
    if unknown_output_dir is None:
        unknown_output_dir = Path(voices_folder).parent / "_unknown"

    report = BatchReport(folders=list(folders))

    for folder in folders:
        folder = Path(folder)
        audio_files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )

        if not audio_files:
            logger.warning("Carpeta %s: sin archivos de audio", folder)
            continue

        for audio_path in audio_files:
            logger.info("Procesando: %s", audio_path)
            try:
                segments = core_analysis(
                    str(audio_path),
                    voices_folder=str(voices_folder),
                    log_folder=str(folder),
                    language=language,
                    modelSize=model_size,
                    ACCESS_TOKEN=access_token,
                    model_type="faster-whisper",
                    skip_enhance=skip_enhance,
                )

                report.processed_files.append(audio_path)

                # Recolectar speakers: conocidos e incógnitos
                unknown_segs: dict[str, list[list[float]]] = {}
                for seg in segments:
                    start, end, text, speaker = seg[0], seg[1], seg[2], seg[3]
                    if speaker == "unknown":
                        # Recuperar tag original no disponible post-transcripción:
                        # usamos el speaker label tal cual viene del pipeline
                        unknown_segs.setdefault(speaker, []).append([start, end])
                    else:
                        report.identified_speakers.add(speaker)

                # Extraer clips de desconocidos
                if unknown_segs:
                    extracted = extract_unknown_speakers(
                        audio_path=audio_path,
                        unknown_segments=unknown_segs,
                        output_dir=unknown_output_dir,
                        min_duration_s=min_unknown_duration_s,
                        max_clips=max_unknown_clips,
                    )
                    for tag, folder_path in extracted.items():
                        report.unknown_speakers.append({
                            "tag": tag,
                            "audio": audio_path,
                            "folder": folder_path,
                        })

            except Exception:
                logger.exception("Error procesando %s", audio_path)
                report.errors += 1

    return report
