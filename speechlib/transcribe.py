import torch
from .whisper_sinhala import (whisper_sinhala)
from faster_whisper import WhisperModel
import whisper

def transcribe(file, language, model_size, whisper_type, quantization):
    res = ""
    if language in ["si", "Si"]:
        res = whisper_sinhala(file)
        return res
    elif model_size in ["base", "tiny", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]:
        if whisper_type == "faster-whisper":
            if torch.cuda.is_available():
                if quantization:
                    model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                else:
                    model = WhisperModel(model_size, device="cuda", compute_type="float16")
            else:
                if quantization:
                    model = WhisperModel(model_size, device="cpu", compute_type="int8")
                else:
                    model = WhisperModel(model_size, device="cpu", compute_type="float32")

            if language in model.supported_languages:
                segments, info = model.transcribe(file, language=language, beam_size=5)

                for segment in segments:
                    res += segment.text + " "
                    
                return res
            else:
                Exception("Language code not supported.\nThese are the supported languages:\n", model.supported_languages)
        else:
            try:
                model = whisper.load_model(model_size)
                result = model.transcribe(file, language=language)
                res = result["text"]

                return res
            except Exception as err:
                print("an error occured while transcribing: ", err)
    else:
        raise Exception("only 'base', 'tiny', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3' models are available.")

