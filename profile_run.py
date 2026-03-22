"""
Runs the full speechlib pipeline on the reference MP3 with step_timer profiling.
Output goes to transcript_samples/output/
"""
import os
import sys

os.environ["SPEECHLIB_PROFILE"]         = "1"
os.environ["SPEECHLIB_PROFILE_KERNELS"] = "0"   # off — evita overhead que contamina tiempos

sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio")

from speechlib import Transcriptor

FILE        = r"C:\workspace\#dev\speechlib\transcript_samples\20260211_123242.mp3"
LOG_FOLDER  = r"C:\workspace\#dev\speechlib\transcript_samples\output"
VOICES      = r"C:\workspace\#dev\speechlib\transcript_samples\voices"
LANGUAGE    = "es"
MODEL_SIZE  = "large-v3"
TOKEN       = os.environ["HF_TOKEN"]

os.makedirs(LOG_FOLDER, exist_ok=True)

t = Transcriptor(
    file=FILE,
    log_folder=LOG_FOLDER,
    language=LANGUAGE,
    modelSize=MODEL_SIZE,
    ACCESS_TOKEN=TOKEN,
    voices_folder=VOICES,
    quantization=False,
)

result = t.faster_whisper()
print(f"\nSegments transcribed: {len(result)}")
