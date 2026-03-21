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
