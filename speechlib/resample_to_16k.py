import torchaudio
from .audio_state import AudioState
from .step_timer import timed

TARGET_SR = 16000


@timed("resample_to_16k")
def resample_to_16k(state: AudioState) -> AudioState:
    waveform, sr = torchaudio.load(str(state.working_path))
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = state.artifacts_dir / "16k.wav"

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    torchaudio.save(str(out_path), waveform, TARGET_SR, bits_per_sample=16)
    return state.model_copy(update={"working_path": out_path, "is_16khz": True})
