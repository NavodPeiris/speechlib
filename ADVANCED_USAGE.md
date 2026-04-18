# Advanced Usage Guide

Speechlib's latest architectural updates introduce deep customizability, making it capable of satisfying even the most stringent enterprise requirements.

## 1. Deep Customizability (`**kwargs` Routing)
You are no longer restricted to just the top-level parameters. The `Transcriptor` object accepts an unlimited number of keyword arguments (`**kwargs`) which are safely routed through the stack.

### Pyannote Diarization Tuning
By passing diarization keys directly into `Transcriptor`, you can heavily constrain the diarization bounds to prevent hallucinations and merge errors:
- `min_speakers` (int): Minimum number of distinct speakers expected in the audio.
- `max_speakers` (int): Maximum limit of distinct speakers.

### Whisper / Faster-Whisper Decoding Strategies
You can control the exact internal decoding logic of your Whisper models by passing any supported Generation Config key:
- `beam_size` (int): Increase this (e.g. `5` to `10`) to massively improve accuracy at the cost of processing speed.
- `temperature` (float | list): Set temperature fallbacks to allow the model to try different sampling techniques if transcription loops or fails. (e.g. `[0.0, 0.2, 0.4]`).
- `condition_on_previous_text` (bool): Turn this `False` to prevent the model from getting stuck in repetitive hallucinations across segment bounds.
- `patience` (float): Beam search patience factor.

### Strict Validation
All parameters passed as `**kwargs` are strictly validated based on your chosen backend. If you pass a Faster-Whisper specific argument (like `beam_size`) and attempt to run `.assemby_ai_model()`, it will immediately throw a highly-descriptive `ValueError` rather than failing silently, ensuring maximum API integrity.

## 2. Advanced Output Formats
Using `output_format="json"` or `output_format="both"` will generate incredibly detailed `.json` logs alongside the standard `.txt`.

**Sample Output Structure (`log_folder/obama_zach_140230_en.json`):**
```json
{
    "file_name": "obama_zach.wav",
    "language_detected": "en",
    "model_used": "large-v3-turbo",
    "segments": [
        {
            "file_name": "obama_zach.wav",
            "start_time": 0.0,
            "end_time": 3.4,
            "text": "Hello, everyone. Thank you for having me.",
            "speaker": "obama",
            "model_used": "large-v3-turbo",
            "language_detected": "en"
        }
    ]
}
```

## 3. High-Performance Batch Processing
Passing a list of strings (`["file1.wav", "file2.wav"]`) to `file` invokes the new batch engine.
- **Pipeline Caching**: The heavy Pyannote pipeline (which can take many seconds to load into memory) is cached globally on the first initialization. Subsequent files process significantly faster.
- **Fault Tolerance**: If one file in the batch throws a decoding error, it prints the explicit exception, skips that specific file, and seamlessly processes the rest of the queue.

## 4. Universal Auto-Detection
If you leave `language=None` (which is now the default):
- **Whisper & Faster-Whisper**: Runs the initial seconds of audio through the token predictor and automatically assigns the detected language token to the generation parameters.
- **AssemblyAI**: Safely maps to their internal `language_detection=True` config instead of breaking.

## 5. Security & Tokens
Never hardcode your access token again. If `ACCESS_TOKEN` is `None` or omitted, Speechlib will check:
1. `os.environ.get("HUGGINGFACE_ACCESS_TOKEN")`
2. `os.environ.get("HF_TOKEN")`
