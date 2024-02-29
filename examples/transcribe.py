from speechlib import Transcriptor

file = "obama_zach.wav"  # your audio file
voices_folder = "voices" # voices folder containing voice samples for recognition
language = "en"          # language code
log_folder = "logs"      # log folder for storing transcripts
modelSize = "tiny"     # size of model to be used [tiny, small, medium, large-v1, large-v2, large-v3]
quantization = False   # setting this 'True' may speed up the process but lower the accuracy

transcriptor = Transcriptor(file, log_folder, language, modelSize, voices_folder, quantization)

res = transcriptor.transcribe()
