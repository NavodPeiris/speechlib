from speechlib import Transcriptor

file = "obama1.wav"  # your audio file
voices_folder = "" # voices folder containing voice samples for recognition
language = "en"          # language code
log_folder = "logs"      # log folder for storing transcripts
modelSize = "tiny"     # size of model to be used [tiny, small, medium, large-v1, large-v2, large-v3]
quantization = False   # setting this 'True' may speed up the process but lower the accuracy
ACCESS_TOKEN = "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" # get permission to access pyannote/speaker-diarization@2.1 on huggingface

# quantization only works on faster-whisper
transcriptor = Transcriptor(file, log_folder, language, modelSize, ACCESS_TOKEN, voices_folder, quantization)

# use normal whisper
#res = transcriptor.whisper()

# use faster-whisper (simply faster)
#res = transcriptor.faster_whisper()

# use a custom trained whisper model
#res = transcriptor.custom_whisper("D:/whisper_tiny_model/tiny.pt")

# use a huggingface whisper model
#res = transcriptor.huggingface_model("Jingmiao/whisper-small-chinese_base")

# use assembly ai model
res = transcriptor.assemby_ai_model("f1d5ee4f7e7a4db4bba1ab4b0a5a1df3")