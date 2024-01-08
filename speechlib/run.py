from speechlib import Transcriptor

file = "example1.wav"
voice_folder = "voices"
language = "sinhala"
log_folder = "logs"

transcriptor = Transcriptor(file, log_folder, language, voice_folder)

res = transcriptor.transcribe()

print("res", res)