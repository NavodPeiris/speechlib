import torchaudio
from .audio_state import AudioState

TARGET_SR = 16000


def resample_to_16k(state: AudioState) -> AudioState:
    waveform, sr = torchaudio.load(str(state.working_path))
    if sr == TARGET_SR:
        return state.model_copy(update={"is_16khz": True})

    resampled = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    out_path = state.working_path.with_stem(state.working_path.stem + "_16k")
    torchaudio.save(str(out_path), resampled, TARGET_SR, bits_per_sample=16)
    return state.model_copy(update={"working_path": out_path, "is_16khz": True})
