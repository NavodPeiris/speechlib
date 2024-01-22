from pydub import AudioSegment
import os

def convert_to_wav(input_file):
    # Load the MP3 file using pydub
    # Check if the file is already in WAV format
    if input_file.lower().endswith(".wav"):
        print(f"{input_file} is already in WAV format.")
        return input_file
    
    audio = AudioSegment.from_file(input_file)

    # Create the output WAV file path
    wav_path = os.path.splitext(input_file)[0] + ".wav"

    # Export the audio to WAV
    audio.export(wav_path, format="wav")

    print(f"{input_file} has been converted to WAV format.")

    return wav_path

