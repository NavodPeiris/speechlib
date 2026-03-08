import wave
import struct
from .audio_state import AudioState


def re_encode(state: AudioState) -> AudioState:
    with wave.open(str(state.working_path), 'rb') as f:
        params = f.getparams()

        if params.sampwidth == 2:
            return state.model_copy(update={"is_16bit": True})

        if params.sampwidth != 1:
            print("Unsupported sample width.")
            return state.model_copy(update={"is_16bit": False})

        frames_8bit = [
            f.readframes(1) for _ in range(params.nframes)
        ]

    out_path = state.working_path.with_stem(state.working_path.stem + "_16bit")

    with wave.open(str(out_path), 'wb') as out:
        out.setparams(params)
        out.setsampwidth(2)
        out.setnchannels(1)
        for sample in frames_8bit:
            value = struct.unpack("<B", sample)[0]
            converted = struct.pack("<h", (value - 128) * 256)
            out.writeframes(converted)

    return state.model_copy(update={"working_path": out_path, "is_16bit": True})
