from transformers import pipeline

def whisper_sinhala(file):
    pipe = pipeline("automatic-speech-recognition", model="Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2")
    res = pipe(file)
    return res["text"]


