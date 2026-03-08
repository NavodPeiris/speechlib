import torchaudio
from pathlib import Path


def slice_and_save(source: str, start_ms: float, end_ms: float, dest: str) -> None:
    waveform, sample_rate = torchaudio.load(source)
    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)
    clip = waveform[:, start_sample:end_sample]
    torchaudio.save(dest, clip, sample_rate, bits_per_sample=16)
