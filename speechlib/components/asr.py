from __future__ import annotations
import os
import tempfile
from typing import BinaryIO
import numpy as np
import torch
import torchaudio
import whisper
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline
import assemblyai as aai
from .base import BaseASR
from ..whisper_sinhala import whisper_sinhala

_model_cache: dict = {}


def _buf_to_numpy(audio: BinaryIO) -> tuple[np.ndarray, int]:
    """Load a WAV buffer to a float32 mono numpy array, resampled to 16 kHz."""
    waveform, sr = torchaudio.load(audio)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze().numpy().astype("float32"), sr

_WHISPER_SIZES = {
    "base", "tiny", "small", "medium",
    "large", "large-v1", "large-v2", "large-v3",
    "turbo", "large-v3-turbo",
}


class WhisperASR(BaseASR):
    """
    OpenAI Whisper backend.

    Parameters
    ----------
    model_size : str
        One of: tiny, base, small, medium, large, large-v1, large-v2,
        large-v3, turbo, large-v3-turbo.
    **kwargs
        Any keyword argument accepted by ``whisper.Whisper.transcribe``
        (e.g. ``beam_size``, ``temperature``, ``patience``,
        ``condition_on_previous_text``).
    """

    def __init__(self, model_size: str = "tiny", **kwargs):
        if model_size not in _WHISPER_SIZES:
            raise ValueError(f"model_size must be one of {sorted(_WHISPER_SIZES)}, got '{model_size}'")
        self.model_size = model_size
        self.kwargs = kwargs

    def _get_model(self):
        key = ("whisper", self.model_size)
        if key not in _model_cache:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model_cache[key] = whisper.load_model(self.model_size, device=device)
        return _model_cache[key]

    def transcribe(self, audio: str | BinaryIO, language: str | None) -> str:
        if language in ("si", "Si"):
            if not isinstance(audio, str):
                arr, sr = _buf_to_numpy(audio)
                return whisper_sinhala({"array": arr, "sampling_rate": sr})
            return whisper_sinhala(audio)
        if not isinstance(audio, str):
            audio, _ = _buf_to_numpy(audio)
        model = self._get_model()
        fp16 = torch.cuda.is_available()
        result = model.transcribe(audio, language=language, fp16=fp16, **self.kwargs)
        return result["text"]


class FasterWhisperASR(BaseASR):
    """
    faster-whisper backend — lower memory, faster inference than OpenAI Whisper.

    Parameters
    ----------
    model_size : str
        One of: tiny, base, small, medium, large, large-v1, large-v2,
        large-v3, turbo, large-v3-turbo.
    quantization : bool
        Use int8 quantization for reduced memory and faster CPU/GPU inference.
    **kwargs
        Any keyword argument accepted by ``WhisperModel.transcribe``
        (e.g. ``beam_size``, ``temperature``, ``patience``,
        ``condition_on_previous_text``).  ``beam_size`` defaults to 5.
    """

    def __init__(self, model_size: str = "tiny", quantization: bool = False, **kwargs):
        if model_size not in _WHISPER_SIZES:
            raise ValueError(f"model_size must be one of {sorted(_WHISPER_SIZES)}, got '{model_size}'")
        self.model_size = model_size
        self.quantization = quantization
        self.kwargs = {"beam_size": 5, **kwargs}

    def _get_model(self) -> WhisperModel:
        key = ("faster-whisper", self.model_size, self.quantization)
        if key not in _model_cache:
            if torch.cuda.is_available():
                compute_type = "int8_float16" if self.quantization else "float16"
                _model_cache[key] = WhisperModel(self.model_size, device="cuda", compute_type=compute_type)
            else:
                compute_type = "int8" if self.quantization else "float32"
                _model_cache[key] = WhisperModel(self.model_size, device="cpu", compute_type=compute_type)
        return _model_cache[key]

    def transcribe(self, audio: str | BinaryIO, language: str | None) -> str:
        if language in ("si", "Si"):
            if not isinstance(audio, str):
                arr, sr = _buf_to_numpy(audio)
                return whisper_sinhala({"array": arr, "sampling_rate": sr})
            return whisper_sinhala(audio)
        if not isinstance(audio, str):
            audio, _ = _buf_to_numpy(audio)
        model = self._get_model()
        if language is not None and language not in model.supported_languages:
            raise ValueError(f"Language '{language}' is not supported by faster-whisper.")
        segments, _ = model.transcribe(audio, language=language, **self.kwargs)
        return "".join(seg.text + " " for seg in segments)


class CustomWhisperASR(BaseASR):
    """
    Local fine-tuned Whisper checkpoint loaded from disk.

    Parameters
    ----------
    model_path : str
        Absolute path to the ``.pt`` model checkpoint file.
    **kwargs
        Any keyword argument accepted by ``whisper.Whisper.transcribe``.
    """

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs

    def _get_model(self):
        key = ("custom", self.model_path)
        if key not in _model_cache:
            model_folder = os.path.dirname(self.model_path) + "/"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model_cache[key] = whisper.load_model(
                self.model_path, download_root=model_folder, device=device
            )
        return _model_cache[key]

    def transcribe(self, audio: str | BinaryIO, language: str | None) -> str:
        if not isinstance(audio, str):
            audio, _ = _buf_to_numpy(audio)
        model = self._get_model()
        fp16 = torch.cuda.is_available()
        result = model.transcribe(audio, language=language, fp16=fp16, **self.kwargs)
        return result["text"]


class HuggingFaceASR(BaseASR):
    """
    Any HuggingFace ``automatic-speech-recognition`` model.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g. ``"openai/whisper-small"``).
    **kwargs
        Extra keyword arguments forwarded to the pipeline ``generate_kwargs``.
    """

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs

    def _get_model(self):
        key = ("huggingface", self.model_id)
        if key not in _model_cache:
            if torch.cuda.is_available():
                device, dtype = 0, torch.float16
            else:
                device, dtype = "cpu", torch.float32
            _model_cache[key] = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device,
                torch_dtype=dtype,
            )
        return _model_cache[key]

    def transcribe(self, audio: str | BinaryIO, language: str | None) -> str:
        if not isinstance(audio, str):
            arr, sr = _buf_to_numpy(audio)
            audio = {"array": arr, "sampling_rate": sr}
        pipe = self._get_model()
        result = pipe(audio, generate_kwargs=self.kwargs) if self.kwargs else pipe(audio)
        return result["text"]


class AssemblyAIASR(BaseASR):
    """
    AssemblyAI cloud transcription.

    Parameters
    ----------
    api_key : str
        AssemblyAI API key.
    speech_model : aai.SpeechModel
        Model variant to use (default ``aai.SpeechModel.nano``).
    """

    def __init__(self, api_key: str, speech_model=None):
        self.api_key = api_key
        self.speech_model = speech_model or aai.SpeechModel.nano

    def transcribe(self, audio: str | BinaryIO, language: str | None) -> str:
        aai.settings.api_key = self.api_key
        config_kwargs: dict = {"speech_model": self.speech_model}
        if language:
            config_kwargs["language_code"] = language
        else:
            config_kwargs["language_detection"] = True
        config = aai.TranscriptionConfig(**config_kwargs)
        if not isinstance(audio, str):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio.read())
                tmp_path = tmp.name
            try:
                transcript = aai.Transcriber(config=config).transcribe(tmp_path)
            finally:
                os.remove(tmp_path)
        else:
            transcript = aai.Transcriber(config=config).transcribe(audio)
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI error: {transcript.error}")
        return transcript.text
