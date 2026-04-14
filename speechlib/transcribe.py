import torch
from .whisper_sinhala import (whisper_sinhala)
from faster_whisper import WhisperModel
import whisper
import os
from transformers import pipeline
import assemblyai as aai

# Cache loaded models to avoid reloading on every segment
_model_cache = {}

def _get_faster_whisper_model(model_size, quantization):
    key = ("faster-whisper", model_size, quantization)
    if key not in _model_cache:
        if torch.cuda.is_available():
            compute_type = "int8_float16" if quantization else "float16"
            _model_cache[key] = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        else:
            compute_type = "int8" if quantization else "float32"
            _model_cache[key] = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    return _model_cache[key]

def _get_whisper_model(model_size):
    key = ("whisper", model_size)
    if key not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[key] = whisper.load_model(model_size, device=device)
    return _model_cache[key]

def _get_custom_whisper_model(custom_model_path):
    key = ("custom", custom_model_path)
    if key not in _model_cache:
        model_folder = os.path.dirname(custom_model_path) + "/"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[key] = whisper.load_model(custom_model_path, download_root=model_folder, device=device)
    return _model_cache[key]

def transcribe(file, language, model_size, model_type, quantization, custom_model_path, hf_model_path, aai_api_key):
    res = ""
    if language in ["si", "Si"]:
        res = whisper_sinhala(file)
        return res
    elif model_size in ["base", "tiny", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]:
        if model_type == "faster-whisper":
            model = _get_faster_whisper_model(model_size, quantization)
            if language in model.supported_languages:
                segments, _ = model.transcribe(file, language=language, beam_size=5)
                for segment in segments:
                    res += segment.text + " "
                return res
            else:
                raise Exception("Language code not supported.\nThese are the supported languages:\n" + str(model.supported_languages))
        elif model_type == "whisper":
            try:
                model = _get_whisper_model(model_size)
                fp16 = torch.cuda.is_available()
                result = model.transcribe(file, language=language, fp16=fp16)
                return result["text"]
            except Exception as err:
                print("an error occured while transcribing: ", err)
        elif model_type == "custom":
            try:
                model = _get_custom_whisper_model(custom_model_path)
                fp16 = torch.cuda.is_available()
                result = model.transcribe(file, language=language, fp16=fp16)
                return result["text"]
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "huggingface":
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                key = ("huggingface", hf_model_path)
                if key not in _model_cache:
                    _model_cache[key] = pipeline("automatic-speech-recognition", model=hf_model_path, device=device)
                pipe = _model_cache[key]
                result = pipe(file)
                return result['text']
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "assemblyAI":
            try:
                aai.settings.api_key = aai_api_key
                config = aai.TranscriptionConfig(
                    speech_model=aai.SpeechModel.nano,
                    language_code=language
                )
                transcriber = aai.Transcriber(config=config)
                transcript = transcriber.transcribe(file)
                if transcript.status == aai.TranscriptStatus.error:
                    print(transcript.error)
                    raise Exception(f"an error occured while transcribing: {transcript.error}")
                else:
                    return transcript.text
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        else:
            raise Exception(f"model_type {model_type} is not supported")
    else:
        raise Exception("only 'base', 'tiny', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3' models are available.")
