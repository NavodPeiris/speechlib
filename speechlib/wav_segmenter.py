import os
from pydub import AudioSegment
from .transcribe import (transcribe)

# segment according to speaker
def wav_file_segmentation(file_name, segments, language, modelSize, model_type, quantization, custom_model_path, hf_model_path, aai_api_key):
    # Load the WAV file
    audio = AudioSegment.from_file(file_name, format="wav")
    trans = ""

    texts = []

    folder_name = "segments"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    i = 0

    for segment in segments:

        start = segment[0] * 1000   # start time in miliseconds
        end = segment[1] * 1000     # end time in miliseconds
        clip = audio[start:end]
        i = i + 1
        file = folder_name + "/" + "segment"+ str(i) + ".wav"
        clip.export(file, format="wav")

        try:
            trans = transcribe(file, language, modelSize, model_type, quantization, custom_model_path, hf_model_path, aai_api_key)  
            
            # return -> [[start time, end time, transcript], [start time, end time, transcript], ..]
            texts.append([segment[0], segment[1], trans])
        except Exception as err:
            print("ERROR while transcribing: ", err)
        # Delete the WAV file after processing
        os.remove(file)

    return texts
