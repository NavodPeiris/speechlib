import torch
import torchaudio
from .audio_state import AudioState
from .step_timer import timed

TARGET_LUFS = -14.0
TRUE_PEAK_DB = -1.0


@timed("loudnorm")
def loudnorm(state: AudioState) -> AudioState:
    waveform, sr = torchaudio.load(str(state.working_path))

    import math
    current_lufs = torchaudio.functional.loudness(waveform, sr).item()

    if not math.isfinite(current_lufs) or current_lufs < -70:
        return state.model_copy(update={"is_normalized": True})

    if abs(current_lufs - TARGET_LUFS) < 0.5:
        return state.model_copy(update={"is_normalized": True})

    gain_db = TARGET_LUFS - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    normalized = waveform * gain_linear

    true_peak = 10 ** (TRUE_PEAK_DB / 20)
    normalized = torch.clamp(normalized, -true_peak, true_peak)

    torchaudio.save(str(state.working_path), normalized, sr, bits_per_sample=16)
    return state.model_copy(update={"is_normalized": True})
