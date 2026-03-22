"""
Runs the full speechlib pipeline on the reference MP3 with step_timer profiling.
Output goes to transcript_samples/output/
"""
import os
import sys
import argparse

os.environ["SPEECHLIB_PROFILE"]         = "1"
os.environ["SPEECHLIB_PROFILE_KERNELS"] = "0"   # off — evita overhead que contamina tiempos

sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio")

from speechlib.core_analysis import core_analysis

parser = argparse.ArgumentParser()
parser.add_argument("--skip-enhance", action="store_true", help="Skip speech enhancement step")
args = parser.parse_args()

FILE        = r"C:\workspace\#dev\speechlib\transcript_samples\20260211_123242.mp3"
LOG_FOLDER  = r"C:\workspace\#dev\speechlib\transcript_samples\output"
VOICES      = r"C:\workspace\#dev\speechlib\transcript_samples\voices"
LANGUAGE    = "es"
MODEL_SIZE  = "large-v3-turbo"
TOKEN       = os.environ["HF_TOKEN"]

os.makedirs(LOG_FOLDER, exist_ok=True)

result = core_analysis(
    FILE, VOICES, LOG_FOLDER, LANGUAGE, MODEL_SIZE, TOKEN,
    "faster-whisper", quantization=False, skip_enhance=args.skip_enhance,
)
print(f"\nSegments transcribed: {len(result)}")
