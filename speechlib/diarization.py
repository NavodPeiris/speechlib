from functools import lru_cache

import torch
from pyannote.audio import Pipeline


@lru_cache(maxsize=1)
def get_diarization_pipeline(token: str):
    """Carga y cachea el pipeline de diarización pyannote/speaker-diarization-3.1."""
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    return pipeline
