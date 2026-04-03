import torchaudio
import torchaudio.functional as F
from pathlib import Path


def slice_and_save(source: str, start_ms: float, end_ms: float, dest: str) -> None:
    waveform, sample_rate = torchaudio.load(source)
    total_samples = waveform.shape[1]
    start_sample = min(int(start_ms * sample_rate / 1000), total_samples)
    end_sample = min(int(end_ms * sample_rate / 1000), total_samples)
    clip = waveform[:, start_sample:end_sample]
    torchaudio.save(dest, clip, sample_rate, bits_per_sample=16)


def extract_audio_segment(
    source: str | Path,
    dest: str | Path,
    start_s: float,
    duration_s: float,
    target_sr: int = 16000,
    mono: bool = True,
) -> Path:
    """Extrae ventana de audio, convierte a mono y resamplea via torchaudio.

    A diferencia de slice_and_save (que toma ms y no resamplea),
    acepta segundos y produce WAV mono 16kHz listo para embeddings.
    No depende de ffmpeg.
    """
    waveform, sr = torchaudio.load(str(source))
    start_sample = int(start_s * sr)
    end_sample = min(int((start_s + duration_s) * sr), waveform.shape[1])
    clip = waveform[:, start_sample:end_sample]
    if mono and clip.shape[0] > 1:
        clip = clip.mean(dim=0, keepdim=True)
    if sr != target_sr:
        clip = F.resample(clip, sr, target_sr)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dest), clip, target_sr, bits_per_sample=16)
    return dest
