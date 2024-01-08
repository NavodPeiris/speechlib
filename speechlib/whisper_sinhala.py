from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2")

def whisper_sinhala(file):
    res = pipe(file)
    return res["text"]


