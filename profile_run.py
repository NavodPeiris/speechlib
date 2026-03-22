"""
Runs the full speechlib pipeline on the reference MP3 with step_timer profiling.
Output VTT is written to the output/ subdirectory relative to the audio file.
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
parser.add_argument("--speaker-grouping", "-g", action="store_true",
                    help="Group by speaker turns only (default: sentence grouping)")
args = parser.parse_args()

FILE        = r"C:\workspace\#dev\speechlib\transcript_samples\20260211_123242.mp3"
VOICES      = r"C:\workspace\#dev\speechlib\transcript_samples\voices"
LANGUAGE    = "es"
MODEL_SIZE  = "large-v3-turbo"
TOKEN       = os.environ["HF_TOKEN"]

grouping_mode = "speaker" if args.speaker_grouping else "sentences"

result = core_analysis(
    FILE, VOICES, None, LANGUAGE, MODEL_SIZE, TOKEN,
    "faster-whisper", quantization=False, skip_enhance=args.skip_enhance,
    output_format="vtt", grouping_mode=grouping_mode,
)
print(f"\nSegments transcribed: {len(result)}")
