from pathlib import Path
from pydantic import BaseModel, ConfigDict


class AudioState(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: Path
    working_path: Path
    is_wav: bool = False
    is_mono: bool = False
    is_16bit: bool = False
    is_16khz: bool = False
    is_normalized: bool = False
    is_enhanced: bool = False

    @property
    def artifacts_dir(self) -> Path:
        """Carpeta oculta junto al source para todos los artefactos del pipeline.

        Ejemplo: /rec/Voz 260320.m4a → /rec/.Voz 260320/
        """
        return self.source_path.parent / f".{self.source_path.stem}"
