import wave
import struct

def re_encode(file_name):

    with wave.open(file_name, 'rb') as original_file:

        # Get the original audio parameters
        params = original_file.getparams()

        # Check if the sample width is already 16-bit
        if params.sampwidth == 2:

            print("The file already has 16-bit samples.")

        elif params.sampwidth == 1:
            
            # Open a new WAV file with 16-bit samples
            file_name = file_name + '_16bit.wav'

            with wave.open(file_name, 'wb') as new_file:
                # Set the new audio parameters
                new_file.setparams(params)
                new_file.setsampwidth(2)
                new_file.setnchannels(1)

                # Read and convert each sample
                for _ in range(params.nframes):
                    sample = original_file.readframes(1)
                    sample_value = struct.unpack("<B", sample)[0]
                    new_sample_value = (sample_value - 128) * 256
                    new_sample = struct.pack("<h", new_sample_value)
                    new_file.writeframes(new_sample)

            print("Conversion completed. Saved as " + file_name)
        else:
            print("Unsupported sample width.")