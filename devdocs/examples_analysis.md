# Examples Analysis

---

## examples/preprocess.py

Demonstrates standalone use of `PreProcessor` to normalize an audio file
before transcription. Runs the three preprocessing steps manually in sequence.

```python
from speechlib import PreProcessor

file = "obama1.mp3"
prep = PreProcessor()
wav_file = prep.convert_to_wav(file)   # MP3 → WAV
prep.convert_to_mono(wav_file)         # stereo → mono
prep.re_encode(wav_file)               # 8-bit → 16-bit PCM
```

### Issues

**Broken chain:** `convert_to_wav` returns the new WAV path, which is correctly
captured in `wav_file`. But `convert_to_mono` and `re_encode` return `None`
(they were in-place before the AudioState refactor). The example doesn't capture
their return values, so it never passes the updated path forward.

With the current refactored API (`AudioState`), the `PreProcessor` public methods
still wrap the old string-based interface (`speechlib.py` L275-283). Internally
they call the refactored functions but discard the returned `AudioState`. The
example works only if the input is already WAV and mono — otherwise the chain
silently produces the wrong file.

**Hardcoded filename:** `"obama1.mp3"` — not a fixture, assumed to exist locally.

### What it should look like after PreProcessor is updated

```python
prep = PreProcessor()
wav_file = prep.convert_to_wav("obama1.mp3")
mono_file = prep.convert_to_mono(wav_file)
encoded_file = prep.re_encode(mono_file)
```

---

## examples/transcribe.py

Demonstrates full transcription pipeline via `Transcriptor`. Shows all five
transcription backends (one active, four commented out).

```python
from speechlib import Transcriptor

transcriptor = Transcriptor(
    file         = "obama_zach.wav",
    log_folder   = "logs",
    language     = "en",
    modelSize    = "tiny",
    ACCESS_TOKEN = "huggingface access token",
    voices_folder= "",
    quantization = False,
)

res = transcriptor.whisper()
# res = transcriptor.faster_whisper()
# res = transcriptor.custom_whisper("D:/whisper_tiny_model/tiny.pt")
# res = transcriptor.huggingface_model("Jingmiao/whisper-small-chinese_base")
# res = transcriptor.assemby_ai_model("assemblyAI api key")
```

### What each backend needs

| Backend | Extra param | Notes |
|---|---|---|
| `whisper()` | — | default, loads model locally |
| `faster_whisper()` | — | faster, int8 quantization supported |
| `custom_whisper(path)` | local `.pt` path | fine-tuned checkpoint |
| `huggingface_model(id)` | HF model ID | any ASR pipeline on HF Hub |
| `assemby_ai_model(key)` | AssemblyAI API key | cloud, no local GPU needed |

### Notes

- `voices_folder = ""` disables speaker recognition — diarization still runs
  but speakers are labeled `SPEAKER_00`, `SPEAKER_01`, etc.
- `ACCESS_TOKEN` must have access to `pyannote/speaker-diarization@2.1` on HuggingFace.
- `quantization=True` only has effect with `faster_whisper()` — ignored by all other backends.
- `res` is never printed or used in the example — the transcript is written to `logs/`.

---

## Summary of gaps in examples

| Gap | File | Impact |
|---|---|---|
| `PreProcessor` chain broken — return values not propagated | `preprocess.py` | silent wrong output |
| `res` unused — user might expect print output | `transcribe.py` | UX confusion |
| `ACCESS_TOKEN` is a literal placeholder string | `transcribe.py` | will fail at runtime |
| Hardcoded filenames not present in repo | both | example not runnable as-is |
