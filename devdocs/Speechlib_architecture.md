# SpeechLib: Architecture & Process Flow

**Version:** 2.5
**Updated:** 2026-03-21

---

## Overview

SpeechLib transcribes audio files with speaker identification. It integrates:

1. **Preprocessing** — normalizes audio to a standard format via AudioState pipeline
2. **Diarization** (Pyannote) — detects who speaks and when
3. **Speaker Recognition** (optional) — maps speaker IDs to real names
4. **Transcription** (multi-backend) — converts speech to text
5. **Output** — writes structured log file

---

## High-Level Architecture

```
User API
  ├── Transcriptor       → core_analysis()
  └── PreProcessor       → individual preprocessing steps

core_analysis()  [orchestrator]
  ├── Preprocessing      (AudioState pipeline)
  ├── Diarization        (Pyannote)
  ├── Speaker Recognition (optional, Pyannote Embedding)
  ├── Segmentation & Transcription
  └── Write Log File

Modules
  ├── audio_state.py         AudioState — pipeline state carrier
  ├── audio_utils.py         slice_and_save — torchaudio-based audio slicing
  ├── convert_to_wav.py      any format → WAV  (torchaudio)
  ├── convert_to_mono.py     stereo → mono     (wave + numpy)
  ├── re_encode.py           8-bit → 16-bit PCM (wave)
  ├── resample_to_16k.py     any SR → 16 kHz   (torchaudio.functional.resample)
  ├── loudnorm.py            EBU R128 → -14 LUFS true peak -1 dB (torchaudio)
  ├── enhance_audio.py       speech enhancement MossFormer2_SE_48K (ClearVoice)
  ├── segment_merger.py      merge_short_turns — post-diarization segment merging
  ├── wav_segmenter.py       slice + transcribe per segment (torchaudio)
  ├── speaker_recognition.py  pyannote embedding + cosine similarity
  ├── transcribe.py          multi-backend transcription (LRU-cached models)
  ├── whisper_sinhala.py     Sinhala-specific HF pipeline
  └── write_log_file.py      transcript → .txt or .srt
```

---

## Preprocessing Pipeline (AudioState)

Each step receives and returns an `AudioState`. The **source file is never modified**.

```python
class AudioState(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path:   Path    # original — immutable
    working_path:  Path    # current processed file
    is_wav:        bool = False
    is_mono:       bool = False
    is_16bit:      bool = False
    is_16khz:      bool = False   # set by resample_to_16k
    is_normalized: bool = False   # set by loudnorm
    is_enhanced:   bool = False   # set by enhance_audio
```

### Flow

```
input: "meeting.mp3"

AudioState(source="meeting.mp3", working="meeting.mp3")
    │
    ▼ convert_to_wav()        torchaudio.load / torchaudio.save
AudioState(working="meeting.wav",            is_wav=True)
    │
    ▼ convert_to_mono()       wave + numpy
AudioState(working="meeting_mono.wav",       is_wav=True, is_mono=True)
    │
    ▼ re_encode()             wave
AudioState(working="meeting_mono_16bit.wav",     is_wav=True, is_mono=True, is_16bit=True)
    │
    ▼ resample_to_16k()       torchaudio.functional.resample
AudioState(working="meeting_mono_16bit_16k.wav", ..., is_16khz=True)
    │
    ▼ loudnorm()              EBU R128 — in-place, -14 LUFS / -1 dBTP
AudioState(working="meeting_mono_16bit_16k.wav", ..., is_normalized=True)
    │
    ▼ enhance_audio()         ClearVoice MossFormer2_SE_48K (GPU, lazy-loaded)
AudioState(working=".../MossFormer2_SE_48K/meeting_mono_16bit_16k.wav", is_enhanced=True)
    │
    ▼ pipeline({"waveform": ..., "sample_rate": ...})   ← diarization
```

### Step behavior

| Step | Condition | Action |
|---|---|---|
| `convert_to_wav` | already `.wav` | `is_wav=True`, working_path unchanged |
| | other format | torchaudio converts to new `.wav`, updates working_path |
| `convert_to_mono` | 1 channel | `is_mono=True`, working_path unchanged |
| | 2+ channels | numpy mix-down to `*_mono.wav`, updates working_path |
| `re_encode` | sampwidth == 2 | `is_16bit=True`, working_path unchanged |
| | sampwidth == 1 | converts to `*_16bit.wav`, updates working_path |
| `resample_to_16k` | already 16 kHz | `is_16khz=True`, working_path unchanged |
| | other SR | torchaudio.functional.resample → `*_16k.wav`, updates working_path |
| `loudnorm` | within ±0.5 LUFS of target | `is_normalized=True`, no file rewrite |
| | LUFS < -70 (silence) | `is_normalized=True`, skip (avoid gain on silence) |
| | other | apply gain → clamp at -1 dBTP → overwrite working_path in-place |
| `enhance_audio` | always | ClearVoice SE → `*_enhanced_out/MossFormer2_SE_48K/*.wav`, updates working_path |

### Wiring in core_analysis

```python
state = AudioState(source_path=Path(file_name), working_path=Path(file_name))
state = convert_to_wav(state)
state = convert_to_mono(state)
state = re_encode(state)
state = resample_to_16k(state)   # Slice A
state = loudnorm(state)           # Slice B — normalize before SE
state = enhance_audio(state)      # Slice D — MossFormer2_SE_48K

# diarization receives waveform dict (pyannote 4.x API)
waveform, sample_rate = torchaudio.load(str(state.working_path))
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```

---

## Full Pipeline Flow

```
Input: audio file (any format)
    │
    ▼ Preprocessing
    │   convert_to_wav → convert_to_mono → re_encode
    │     → resample_to_16k → loudnorm → enhance_audio
    │   Result: WAV, mono, 16-bit, 16 kHz, -14 LUFS, SE-enhanced
    │
    ▼ Diarization — Pyannote 4.x (speaker-diarization-3.1)
    │   Input:  {"waveform": tensor, "sample_rate": int}
    │   Output: [[start, end, "SPEAKER_00"], ...]
    │   Note:   pyannote 4.x may return a DiarizeOutput object;
    │           core_analysis reads .speaker_diarization if present,
    │           else uses the object directly (backwards-compat shim)
    │
    ▼ Speaker Recognition — optional
    │   Input:  segments + voices_folder
    │   Model:  pyannote/embedding (downloaded to cache)
    │   Output: {"SPEAKER_00": "john_doe", ...}
    │
    ▼ Segmentation & Transcription
    │   For each speaker's segments:
    │     audio_utils.slice_and_save (torchaudio) → temp WAV → transcribe() → delete temp
    │   Output: [[start, end, transcript], ...]
    │
    ▼ Segment Merging (optional)
    │   merge_short_turns(common, max_gap_s=0.5)
    │   Merges consecutive same-speaker segments with gap < max_gap_s
    │
    ▼ Write Log File
        TXT: {log_folder}/{filename}_{HHMMSS}_{lang}.txt
        SRT: {log_folder}/{filename}_{HHMMSS}_{lang}.srt  (optional)
```

---

## Data Structure Evolution

```
After diarization:
  common = [[0.0, 2.3, "SPEAKER_00"],
            [2.3, 5.1, "SPEAKER_01"], ...]

After speaker recognition (if enabled):
  common = [[0.0, 2.3, "john_doe"],
            [2.3, 5.1, "jane_smith"], ...]

After segment merging (optional):
  common = [[0.0, 5.1, "john_doe"], ...]  ← merged if gap < 0.5s and same speaker

Return value of core_analysis:
  [[0.0, 2.3, "Hello everyone",    "john_doe"],
   [2.3, 5.1, "Hi, glad to be here", "jane_smith"], ...]

TXT log file:
  john_doe (0.0 : 2.3) : Hello everyone
  jane_smith (2.3 : 5.1) : Hi, glad to be here

SRT log file:
  1
  00:00:00,000 --> 00:00:02,300
  john_doe: Hello everyone

  2
  00:00:02,300 --> 00:00:05,100
  jane_smith: Hi, glad to be here
```

---

## Transcription Backends

| `model_type` | Backend | Notes |
|---|---|---|
| `"whisper"` | OpenAI Whisper | loads model per call |
| `"faster-whisper"` | CTranslate2/faster-whisper | `BatchedInferencePipeline(batch_size=16)`; model cached via `@lru_cache(maxsize=4)`; **5.12x speedup** medido en RTX 2070 Super |
| `"custom"` | local Whisper checkpoint | path via `custom_model_path` |
| `"huggingface"` | HF ASR pipeline | model ID via `hf_model_id` |
| `"assemblyAI"` | AssemblyAI cloud API | requires `aai_api_key` |

**Special case:** language `"si"` (Sinhala) bypasses `model_type` and always uses `whisper_sinhala.py` → HF model `Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2`.

---

## Temporary Files

| Location | Created by | Deleted by |
|---|---|---|
| `segments/segment_N.wav` | `wav_segmenter.py` | `wav_segmenter.py` (after transcription) |
| `temp/{name}_segmentN.wav` | `speaker_recognition.py` | `speaker_recognition.py` (after scoring) |
| Cache | Pyannote models at import | never — cached model weights |

---

## User-Facing API

**`Transcriptor(file, log_folder, language, modelSize, ACCESS_TOKEN, voices_folder, quantization)`**
Stores parameters, delegates to `core_analysis()`.
Methods: `whisper()`, `faster_whisper()`, `custom_whisper(path)`, `huggingface_model(id)`, `assemby_ai_model(key)`

**`PreProcessor`**
Exposes preprocessing steps individually.
Methods: `convert_to_wav(file)`, `convert_to_mono(file)`, `re_encode(file)`

---

## Extension Points

### New transcription backend

Add `elif` branch in `speechlib/transcribe.py`, expose method in `Transcriptor`.

### New preprocessing step

```python
# speechlib/my_step.py
def my_step(state: AudioState) -> AudioState:
    ...
    return state.model_copy(update={"working_path": new_path})
```

Wire after `re_encode` in `core_analysis.py`:
```python
state = my_step(state)
```

---

## Known Issues / In Progress

| Issue | Plan |
|---|---|
| `whisper_sinhala.py` has no unit tests | Add mocked unit tests per TDD methodology |

---

## Technology Stack

| Library | Use |
|---|---|
| `pydantic` | AudioState model (v2, ConfigDict) |
| `torchaudio` | format conversion, waveform loading, audio slicing |
| `torch` | device management, tensor ops |
| `wave` | WAV read/write (mono conversion, re-encoding) |
| `numpy` | stereo→mono mix-down |
| `pyannote.audio` 4.x | speaker diarization (`speaker-diarization-3.1`, `token=` auth) and embedding extraction |
| `whisper` / `faster_whisper` | transcription (faster-whisper with `BatchedInferencePipeline`, LRU-cached `WhisperModel`) |
| `ClearVoice` (local) | speech enhancement — `MossFormer2_SE_48K` via `c:\workspace\#dev\ClearerVoice-Studio` |
| `transformers` | HuggingFace ASR pipeline |
| `scipy` | cosine similarity for speaker matching |
| `assemblyai` | cloud transcription API |
