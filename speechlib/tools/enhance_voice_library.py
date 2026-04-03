"""
Procesa WAVs de una librería de voces a través de MossFormer2 (enhance).

Crea _enhanced/ dentro de cada carpeta de speaker con las versiones enhanced.
Salta archivos que ya existen en _enhanced/.

Uso:
    python -m speechlib.tools.enhance_voice_library VOICES_FOLDER

Ejemplo:
    python -m speechlib.tools.enhance_voice_library transcript_samples/voices
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")

from speechlib.speaker_recognition import VOICES_SKIP_PREFIX


_clearvoice_model = None


def _get_clearvoice():
    global _clearvoice_model
    if _clearvoice_model is None:
        from clearvoice import ClearVoice

        _clearvoice_model = ClearVoice(
            task="speech_enhancement",
            model_names=["MossFormer2_SE_48K"],
        )
    return _clearvoice_model


def enhance_wav(src: Path, dst: Path) -> Path:
    """Procesa un WAV por MossFormer2 y guarda en dst."""
    model = _get_clearvoice()
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = model(input_path=str(src), online_write=False)
    model.write(result, output_path=str(dst))
    return dst


def enhance_voice_library(voices_folder: Path) -> None:
    """Genera _enhanced/ para cada speaker en voices_folder."""
    voices_folder = Path(voices_folder)
    for speaker_dir in sorted(voices_folder.iterdir()):
        if not speaker_dir.is_dir() or speaker_dir.name.startswith(VOICES_SKIP_PREFIX):
            continue
        enh_dir = speaker_dir / "_enhanced"
        for wav in sorted(speaker_dir.glob("*.wav")):
            dst = enh_dir / wav.name
            if dst.exists():
                continue
            print(f"  Enhancing {speaker_dir.name}/{wav.name}...")
            enhance_wav(wav, dst)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    voices_folder = Path(sys.argv[1])
    if not voices_folder.is_dir():
        print(f"Error: {voices_folder} no es un directorio")
        sys.exit(1)

    print(f"Enhancing voice library: {voices_folder}")
    enhance_voice_library(voices_folder)
    print("Done.")


if __name__ == "__main__":
    main()
