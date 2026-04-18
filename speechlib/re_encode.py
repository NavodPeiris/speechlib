import wave
import numpy as np

def re_encode(file_name, verbose=False):

    with wave.open(file_name, 'rb') as original_file:

        params = original_file.getparams()

        if params.sampwidth == 2:
            if verbose:
                print("The file already has 16-bit samples.")
            return

        elif params.sampwidth == 1:
            raw = original_file.readframes(params.nframes)
            # Convert unsigned 8-bit [0,255] to signed 16-bit [-32768,32767]
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
            samples = (samples - 128) * 256
            new_frames = samples.astype(np.int16).tobytes()

        else:
            if verbose:
                print("Unsupported sample width.")
            return

    # Write converted data back to the same file
    with wave.open(file_name, 'wb') as new_file:
        new_file.setparams((1, 2, params.framerate, params.nframes, params.comptype, params.compname))
        new_file.writeframes(new_frames)

    if verbose:
        print(f"Conversion completed. Re-encoded {file_name} to 16-bit PCM.")
