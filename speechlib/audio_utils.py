import torchaudio
from pathlib import Path


def slice_and_save(source: str, start_ms: float, end_ms: float, dest: str) -> None:
    waveform, sample_rate = torchaudio.load(source)
    total_samples = waveform.shape[1]
    start_sample = min(int(start_ms * sample_rate / 1000), total_samples)
    end_sample = min(int(end_ms * sample_rate / 1000), total_samples)
    clip = waveform[:, start_sample:end_sample]
    torchaudio.save(dest, clip, sample_rate, bits_per_sample=16)
