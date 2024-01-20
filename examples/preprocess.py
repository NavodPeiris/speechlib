from speechlib import PreProcessor

file = "obama1.mp3"

# convert mp3 to wav
PreProcessor.mp3_to_wav(file)   

wav_file = "obama1.wav"

# convert wav file from stereo to mono
PreProcessor.convert_to_mono(wav_file)

# re-encode wav file to have 16-bit PCM encoding
PreProcessor.re_encode(wav_file)