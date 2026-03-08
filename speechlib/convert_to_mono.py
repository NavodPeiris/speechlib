import wave
import numpy as np
from .audio_state import AudioState


def convert_to_mono(state: AudioState) -> AudioState:
    with wave.open(str(state.working_path), 'rb') as f:
        params = f.getparams()

        if params.nchannels == 1:
            return state.model_copy(update={"is_mono": True})

        frames = f.readframes(-1)

    audio_data = np.frombuffer(frames, dtype=np.int16)
    mono_data = np.mean(audio_data.reshape(-1, params.nchannels), axis=1).astype(np.int16)

    mono_path = state.working_path.with_stem(state.working_path.stem + "_mono")

    with wave.open(str(mono_path), 'wb') as out:
        out.setparams((1, params.sampwidth, params.framerate, len(mono_data), params.comptype, params.compname))
        out.writeframes(mono_data.tobytes())

    return state.model_copy(update={"working_path": mono_path, "is_mono": True})
