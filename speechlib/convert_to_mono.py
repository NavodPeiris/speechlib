import wave
import numpy as np

def convert_to_mono(input_wav):
    # Open the input WAV file
    with wave.open(input_wav, 'rb') as input_file:
        # Get the parameters of the input file
        params = input_file.getparams()

        # Check if the file is stereo
        if params.nchannels > 1:
            # Read the audio data
            frames = input_file.readframes(-1)
            audio_data = np.frombuffer(frames, dtype=np.int16)

            # Take the average of the channels to convert to mono
            mono_audio_data = np.mean(audio_data.reshape(-1, params.nchannels), axis=1)

            # Create a new WAV file for mono audio
            with wave.open(input_wav, 'wb') as output_file:
                # Set the parameters for the output file
                output_file.setparams((1, params.sampwidth, params.framerate, len(mono_audio_data), params.comptype, params.compname))

                # Write the mono audio data to the output file
                output_file.writeframes(np.int16(mono_audio_data))

            print(f'{input_wav} converted to mono')
        else:
            print(f'{input_wav} is already a mono audio file.')


