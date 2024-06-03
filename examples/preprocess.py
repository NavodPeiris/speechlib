from speechlib import PreProcessor

file = "obama1.mp3"

#initialize
prep = PreProcessor()

# convert mp3 to wav
wav_file = prep.convert_to_wav(file)   

# convert wav file from stereo to mono
prep.convert_to_mono(wav_file)

# re-encode wav file to have 16-bit PCM encoding
prep.re_encode(wav_file)