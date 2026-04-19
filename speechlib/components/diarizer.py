from __future__ import annotations
import os
import torch
import torchaudio
from pyannote.audio import Pipeline as PyannotePipeline
from .base import BaseDiarizer

_pipeline_cache: dict = {}

_original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)

torch.load = patched_load

class PyAnnoteDiarizer(BaseDiarizer):
    """
    Speaker diarization via ``pyannote/speaker-diarization@2.1``.

    Parameters
    ----------
    access_token : str
        HuggingFace access token with access to the pyannote model.
    num_speakers : int | None
        Exact number of speakers when known. Overrides ``min_speakers`` /
        ``max_speakers`` when set.
    min_speakers : int
        Minimum number of speakers to detect (default 1). Ignored when
        ``num_speakers`` is set.
    max_speakers : int
        Maximum number of speakers to detect (default 10). Ignored when
        ``num_speakers`` is set.
    """

    def __init__(
        self,
        access_token: str,
        num_speakers: int | None = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
    ):
        self.access_token = access_token
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def _get_pipeline(self) -> PyannotePipeline:
        global _pipeline_cache
        if "pyannote-diarize" not in _pipeline_cache:
            print("Loading pyannote diarization pipeline...")
            pipe = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=self.access_token,
            )
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            else:
                device = torch.device("cpu")
            pipe.to(device)
            _pipeline_cache["pyannote-diarize"] = pipe
        return _pipeline_cache["pyannote-diarize"]

    def diarize(self, file: str) -> list[tuple[float, float, str]]:
        waveform, sample_rate = torchaudio.load(file)
        pipeline = self._get_pipeline()
        if self.num_speakers is not None:
            speaker_kwargs = {"num_speakers": self.num_speakers}
        else:
            speaker_kwargs = {"min_speakers": self.min_speakers, "max_speakers": self.max_speakers}
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **speaker_kwargs,
        )
        return [
            (round(turn.start, 1), round(turn.end, 1), speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
