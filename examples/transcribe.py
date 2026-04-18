import os
from speechlib import Transcriptor

# Make sure you set your HuggingFace token in the environment!
# os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "your_huggingface_token"

# ==========================================
# Example 1: Basic Single File & Auto-Detect
# ==========================================
print("--- Example 1: Basic Usage ---")
basic_transcriptor = Transcriptor(
    file="obama_zach.wav", 
    log_folder="logs_basic", 
    language=None,             # None automatically triggers language detection
    modelSize="turbo",         # 'turbo' models are fully supported!
    # ACCESS_TOKEN=None,       # If omitted, reads from os.environ["HUGGINGFACE_ACCESS_TOKEN"]
    output_format="both",      # Choose between "txt", "json", or "both"
)

# res1 = basic_transcriptor.faster_whisper()


# ==========================================
# Example 2: Batch Processing & Customization
# ==========================================
print("\n--- Example 2: Batch Processing & Advanced Customization ---")
files_to_process = ["obama_zach.wav", "another_audio.wav"]
voices_folder = "" # Folder containing subfolders named after each speaker with voice samples

# You can pass ANY **kwargs to deeply customize Pyannote and Whisper parameters!
advanced_transcriptor = Transcriptor(
    file=files_to_process,     # Pass a list of files! The Pyannote pipeline is automatically cached to prevent reloading!
    log_folder="logs_batch", 
    language="en", 
    modelSize="large-v3-turbo", 
    quantization=True,         # Speeds up faster-whisper via int8 quantization
    output_format="json",      # Output ONLY the rich JSON log
    
    # --- Diarization Kwargs ---
    min_speakers=1,
    max_speakers=5,
    
    # --- Whisper / Faster-Whisper Kwargs ---
    beam_size=10,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    patience=1.0,
    condition_on_previous_text=False
)

# Uncomment the backend you want to use:

# 1. Use normal whisper
# res = advanced_transcriptor.whisper()

# 2. Use faster-whisper (simply faster)
# res = advanced_transcriptor.faster_whisper()

# 3. Use a custom trained whisper model
# res = advanced_transcriptor.custom_whisper("D:/whisper_tiny_model/tiny.pt")

# 4. Use a huggingface whisper model
# res = advanced_transcriptor.huggingface_model("Jingmiao/whisper-small-chinese_base")

# 5. Use assembly ai model
# res = advanced_transcriptor.assemby_ai_model("assemblyAI api key")
