import torch
from .whisper_sinhala import (whisper_sinhala)
from faster_whisper import WhisperModel, BatchedInferencePipeline
import whisper
import os
from transformers import pipeline
import assemblyai as aai
from functools import lru_cache


@lru_cache(maxsize=4)
def _get_faster_whisper_model(model_size: str, device: str, compute_type: str) -> "WhisperModel":
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(file, language, model_size, model_type, quantization, custom_model_path, hf_model_path, aai_api_key):
    res = ""
    if language in ["si", "Si"]:
        res = whisper_sinhala(file)
        return res
    elif model_size in ["base", "tiny", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]:
        if model_type == "faster-whisper":
            if torch.cuda.is_available():
                compute_type = "int8_float16" if quantization else "float16"
                model = _get_faster_whisper_model(model_size, "cuda", compute_type)
            else:
                compute_type = "int8" if quantization else "float32"
                model = _get_faster_whisper_model(model_size, "cpu", compute_type)

            if language in model.supported_languages:
                batched = BatchedInferencePipeline(model=model)
                segments, info = batched.transcribe(
                    file,
                    language=language,
                    beam_size=5,
                    batch_size=16,
                )

                for segment in segments:
                    res += segment.text + " "

                return res
            else:
                Exception("Language code not supported.\nThese are the supported languages:\n", model.supported_languages)
        elif model_type == "whisper":
            try:
                if torch.cuda.is_available():
                    model = whisper.load_model(model_size, device="cuda")
                    result = model.transcribe(file, language=language, fp16=True)
                    res = result["text"]
                else:
                    model = whisper.load_model(model_size, device="cpu")
                    result = model.transcribe(file, language=language, fp16=False)
                    res = result["text"]

                return res
            except Exception as err:
                print("an error occured while transcribing: ", err)
        elif model_type == "custom":
            model_folder = os.path.dirname(custom_model_path)
            model_folder = model_folder + "/"
            print("model file: ", custom_model_path)
            print("model fodler: ", model_folder)
            try:
                if torch.cuda.is_available():
                    model = whisper.load_model(custom_model_path, download_root=model_folder, device="cuda")
                    result = model.transcribe(file, language=language, fp16=True)
                    res = result["text"]
                else:
                    model = whisper.load_model(custom_model_path, download_root=model_folder, device="cpu")
                    result = model.transcribe(file, language=language, fp16=False)
                    res = result["text"]

                return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "huggingface":
            try:
                if torch.cuda.is_available():
                    pipe = pipeline("automatic-speech-recognition", model=hf_model_path, device="cuda")
                    result = pipe(file)
                    res = result['text']
                else:
                    pipe = pipeline("automatic-speech-recognition", model=hf_model_path, device="cpu")
                    result = pipe(file)
                    res = result['text']
                return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "assemblyAI":
            try:
                # Replace with your API key
                aai.settings.api_key = aai_api_key

                # You can set additional parameters for the transcription
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
                    res = transcript.text
                    return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        else:
            raise Exception(f"model_type {model_type} is not supported")
    else:
        raise Exception("only 'base', 'tiny', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3' models are available.")

