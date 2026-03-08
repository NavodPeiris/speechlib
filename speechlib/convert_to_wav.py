from pydub import AudioSegment
from .audio_state import AudioState


def convert_to_wav(state: AudioState) -> AudioState:
    if state.working_path.suffix.lower() == ".wav":
        return state.model_copy(update={"is_wav": True})

    wav_path = state.working_path.with_suffix(".wav")
    audio = AudioSegment.from_file(str(state.working_path))
    audio.export(str(wav_path), format="wav")

    return state.model_copy(update={"working_path": wav_path, "is_wav": True})
