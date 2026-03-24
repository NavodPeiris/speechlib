"""
Procesa grabaciones reales tomando una muestra (min 1-5) de cada archivo.
Identifica speakers conocidos y extrae desconocidos a voices/_unknown/.

Requiere HF_TOKEN en el entorno.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")

from speechlib.batch_process import batch_process

VOICES  = Path(r"C:\workspace\#dev\speechlib\transcript_samples\voices")
UNKNOWN = Path(r"C:\workspace\#dev\speechlib\transcript_samples\voices\_unknown")
TOKEN   = os.environ["HF_TOKEN"]

SAMPLE_START_S = 60    # minuto 1
SAMPLE_END_S   = 300   # minuto 5

FOLDERS = [
    Path(r"C:\workspace\@recordings\20260320 Patricio Renner"),
    Path(r"C:\workspace\@recordings\20260318 Ina TRE"),
]

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".opus"}


def extract_sample(audio_path: Path, start_s: int, end_s: int, dest: Path) -> Path:
    """Extrae [start_s, end_s] del audio usando ffmpeg → WAV mono 16kHz."""
    duration = end_s - start_s
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-t",  str(duration),
        "-i",  str(audio_path),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dest),
    ], capture_output=True, check=True)
    return dest


# Crear muestras en directorio temporal
tmp_dir = Path(tempfile.mkdtemp(prefix="speechlib_samples_"))
print(f"Muestras en: {tmp_dir}")
print(f"Rango: {SAMPLE_START_S//60}min -> {SAMPLE_END_S//60}min\n")

sample_folders: list[Path] = []

for folder in FOLDERS:
    audio_files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"[SKIP] {folder.name}: sin audio")
        continue

    # Tomar solo el primer audio de cada carpeta
    src = audio_files[0]
    sample_dir = tmp_dir / folder.name
    sample_dir.mkdir()
    dest = sample_dir / (src.stem + "_sample.wav")

    print(f"Extrayendo muestra: {src.name} -> {dest.name}")
    extract_sample(src, SAMPLE_START_S, SAMPLE_END_S, dest)
    print(f"  OK ({dest.stat().st_size // 1024} KB)")
    sample_folders.append(sample_dir)

print()

report = batch_process(
    folders=sample_folders,
    voices_folder=VOICES,
    language="es",
    access_token=TOKEN,
    unknown_output_dir=UNKNOWN,
    skip_enhance=True,
    max_unknown_clips=5,
    min_unknown_duration_s=2.0,
)

report.print_summary()

# Limpiar muestras temporales
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
