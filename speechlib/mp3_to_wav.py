from pydub import AudioSegment
import os

def mp3_to_wav(input_file):
    # Load the MP3 file using pydub
    audio = AudioSegment.from_mp3(input_file)

    # Create the output WAV file path
    wav_path = os.path.splitext(input_file)[0] + ".wav"

    # Export the audio to WAV
    audio.export(wav_path, format="wav")

