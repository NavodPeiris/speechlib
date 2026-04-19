import os
import tempfile
from dotenv import load_dotenv
from speechlib import (
    Pipeline,
    PyAnnoteDiarizer,
    SpeechBrainRecognizer,
    FasterWhisperASR,
    WhisperASR,
    HuggingFaceASR,
    AssemblyAIASR,
    BaseASR
)

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

audio_file_path = "obama_zach_short.wav"


# ==========================================
# Example 1: Minimal — diarization + ASR only
# ==========================================
print("--- Example 1: Minimal ---")
pipeline = Pipeline(
    diarization_model=PyAnnoteDiarizer(
        access_token=HF_TOKEN,
        min_speakers=1,
        max_speakers=2,
    ),
    asr_model=FasterWhisperASR("tiny"),
    language=None,       # auto-detect
    log_folder="logs",
    output_format="both",
)

segments = pipeline.run(audio_file_path)


'''
# ==========================================
# Example 2: Minimal — diarization with pre-known exact speaker count + ASR only
# ==========================================
print("--- Example 2: Minimal — diarization with pre-known exact speaker count ---")

pipeline = Pipeline(
    diarization_model=PyAnnoteDiarizer(
        access_token=HF_TOKEN,
        num_speakers=2
    ),
    asr_model=FasterWhisperASR("tiny"),
    language=None,       # auto-detect
    log_folder="logs",
    output_format="txt",
)

segments = pipeline.run(audio_file_path)
'''

'''
# ==========================================
# Example 3: With speaker recognition + SRT
# ==========================================
print("\n--- Example 3: Speaker recognition, SRT ---")

pipeline = Pipeline(
    diarization_model=PyAnnoteDiarizer(
        access_token=HF_TOKEN,
        min_speakers=1,
        max_speakers=2,
    ),
    speaker_recognition_model=SpeechBrainRecognizer(
        "speechbrain/spkrec-ecapa-voxceleb"
    ),
    asr_model=FasterWhisperASR(
        "tiny",
        quantization=True,
        beam_size=5,
        condition_on_previous_text=False,
    ),
    language="en",
    voices_folder="voices",   # subfolders named after each speaker
    log_folder="logs",
    output_format="json",
    srt=True,
    verbose=True,
)

segments = pipeline.run(audio_file_path)
'''

'''
# ==========================================
# Example 4: Batch processing
# ==========================================
print("\n--- Example 4: Batch ---")

pipeline = Pipeline(
    diarization_model=PyAnnoteDiarizer(
        access_token=HF_TOKEN,
        min_speakers=1,
        max_speakers=2,
    ),
    speaker_recognition_model=SpeechBrainRecognizer(
        "speechbrain/spkrec-ecapa-voxceleb"
    ),
    asr_model=FasterWhisperASR(
        "tiny",
        quantization=True,
        beam_size=5,
        condition_on_previous_text=False,
    ),
    language="en",
    voices_folder="voices",   # subfolders named after each speaker
    log_folder="logs",
    output_format="json",
    srt=True,
    verbose=True,
)

batch_results = pipeline.run([audio_file_path, "obama1.wav"])
'''

'''
# ==========================================
# Example 5: OpenAI Whisper backend
# ==========================================
print("\n--- Example 5: OpenAI Whisper ---")

pipeline = Pipeline(
    diarization_model=PyAnnoteDiarizer(
        access_token=HF_TOKEN,
        min_speakers=1,
        max_speakers=2,
    ),
    asr_model=WhisperASR("tiny", temperature=0.0),
    log_folder="logs",
    output_format="txt",
)
pipeline.run(audio_file_path)
'''

'''
# ==========================================
# Example 6: HuggingFace ASR model
# ==========================================
print("\n--- Example 6: HuggingFace ASR ---")

pipeline = Pipeline(
    diarization_model=PyAnnoteDiarizer(
        access_token=HF_TOKEN,
        min_speakers=1,
        max_speakers=2,
    ),
    asr_model=HuggingFaceASR("distil-whisper/distil-small.en"),
    language="en",
    log_folder="logs",
    output_format="json",
)
pipeline.run(audio_file_path)
'''

'''
# ==========================================
# Example 7: AssemblyAI ASR model
# ==========================================
print("\n--- Example 7: AssemblyAI ---")

pipeline = Pipeline(
    diarization_model=PyAnnoteDiarizer(
        access_token=HF_TOKEN,
        min_speakers=1,
        max_speakers=2,
    ),
    asr_model=AssemblyAIASR(api_key=os.environ.get("AAI_KEY")),
    log_folder="logs",
    output_format="json",
)
pipeline.run(audio_file_path)
'''

'''
# ==========================================
# Example 8: custom ASR stage
# ==========================================
print("\n--- Example 8: custom ASR stage ---")
import nemo.collections.asr as nemo_asr
import threading

class NemoASR(BaseASR):
    def __init__(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
        self.model.freeze()
        self._lock = threading.Lock()   # this model does not support parallelism unfortunately

    def transcribe(self, audio, language):
        with self._lock:
            if not isinstance(audio, str):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.read())
                    tmp_path = tmp.name
                try:
                    output = self.model.transcribe([tmp_path], timestamps=False)
                finally:
                    os.remove(tmp_path)
            else:
                output = self.model.transcribe([audio], timestamps=False)

        return output[0].text

pipeline = Pipeline(
    diarization_model=PyAnnoteDiarizer(
        access_token=HF_TOKEN,
        num_speakers=2
    ),
    asr_model=NemoASR(),
    log_folder="logs",
    output_format="json",
)

pipeline.run(audio_file_path)
'''