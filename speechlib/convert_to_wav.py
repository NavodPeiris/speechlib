import torchaudio
from .audio_state import AudioState


def convert_to_wav(state: AudioState) -> AudioState:
    if state.working_path.suffix.lower() == ".wav":
        return state.model_copy(update={"is_wav": True})

    wav_path = state.working_path.with_suffix(".wav")

    waveform, sample_rate = torchaudio.load(str(state.working_path))
    torchaudio.save(str(wav_path), waveform, sample_rate)

    return state.model_copy(update={"working_path": wav_path, "is_wav": True})
