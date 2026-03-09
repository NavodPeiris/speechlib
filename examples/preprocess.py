from speechlib import PreProcessor

file = "obama1.mp3"
prep = PreProcessor()

wav_file = prep.convert_to_wav(file)
mono_file = prep.convert_to_mono(wav_file)
enc_file = prep.re_encode(mono_file)

print(f"Final processed file: {enc_file}")
