# Session: Speaker Identification & Batch Processing

## Objective

Extend `speechlib` pipeline to process multiple real audio recordings, identify known speakers against a voice library, and extract unknown speakers as WAV clips for later naming and re-ingestion. Target recordings: two folders with Agustin (known) + unknown persons.

---

## Modules Implemented

### `speechlib/speaker_recognition.py` — `find_best_speaker()`
- Added `threshold: float` parameter to cosine similarity gate.
- Returns `"unknown"` when `best_score < threshold` instead of returning best match regardless of confidence.
- Skips voice library folders starting with `_` to prevent `_unknown/` subdirs from being scanned as speaker profiles.

### `speechlib/extract_unknown_speakers.py`
- Input: `{speaker_tag: [[start_s, end_s], ...]}`, audio path, output dir.
- Filters segments by `min_duration_s`, selects `max_clips` longest, saves to `output_dir/{tag}_{audio_stem}/segment_NN.wav`.
- Returns `{speaker_tag: Path}`.

### `speechlib/batch_process.py`
- Iterates folders → audio files → calls `core_analysis()` per file.
- Aggregates known speakers (`identified_speakers: set[str]`) and unknown clips.
- Calls `extract_unknown_speakers()` for segments tagged `"unknown"`.
- Returns `BatchReport` dataclass with `.print_summary()`.

### `process_recordings.py`
- Script: extracts 4-minute sample (min 1–5) per recording via `ffmpeg → WAV mono 16kHz` into a temp dir.
- Calls `batch_process()` with `skip_enhance=True`.
- Cleans temp dir after run.

---

## Threshold Calibration

**Finding:** `pyannote/embedding` cosine similarity for same speaker across different recording conditions (mobile phone recordings vs. controlled voice library) ranges **0.35–0.50**, not 0.75–0.95.

**Empirical data from real recordings:**
| Condition | Score range |
|---|---|
| Agustin (known, same speaker) | 0.35 – 0.504 |
| Jolyon (false positive on Patricio segments) | 0.338 – 0.398 |
| Unknown women (Ina TRE recording) | 0.060 – 0.270 |

**Decision:** threshold = `0.40`
- Catches Agustin (max 0.504, several segments > 0.40).
- Eliminates Jolyon false positive (max 0.398 < 0.40).
- Rejects all unknown voices (all scores < 0.30).

---

## What Worked

- `find_best_speaker` threshold gate: blocks false positives, correct abstention on unknown voices.
- `_` prefix exclusion in `speaker_recognition`: eliminates embedding errors from stale `_unknown/` entries.
- `extract_unknown_speakers`: produces correctly named, duration-filtered WAV clips.
- `batch_process` + `BatchReport`: correct aggregation across multiple folders/files.
- ffmpeg sample extraction (min 1–5): avoids processing 135-min files; reduces per-run time from 40+ min to ~3 min.
- `skip_enhance=True`: appropriate for speaker identification task; enhance is only needed for transcription quality.
- ATDD coverage: threshold tests, extract_unknown_speakers tests, batch_process tests all pass.

---

## What Did Not Work

### Initial threshold 0.75 — complete failure
- All known speakers rejected. `identified_speakers = []`.
- Root cause: assumed 0.75 was a standard pyannote threshold; actual cross-condition similarity is ~0.5 max.
- **Fix:** empirical calibration via debug print of all scores per segment, then set threshold = 0.40.

### `_unknown/` inside `voices/` scanned as speaker
- `speaker_recognition` iterates all subdirs of `voices_folder` including `_unknown/`.
- Subdirs inside `_unknown/` (clip folders) are paths, not WAV files → `pyannote inference()` throws "File does not exist".
- **Fix:** skip dirs starting with `_` in speaker_recognition.

### Unknown speakers merged under single `"unknown"` tag
- `core_analysis` replaces all unidentified diarization tags with the string `"unknown"` before returning segments.
- `batch_process` groups all `speaker == "unknown"` segments under one key → `extract_unknown_speakers` receives mixed clips from different unknown speakers in one bucket.
- **Consequence:** in Ina TRE recording, clips from multiple women were merged into one folder (`unknown_Voz 260318_190843_sample/`).
- **Status:** not fixed. Requires `core_analysis` to preserve original diarization tag (e.g., `SPEAKER_01`) for unidentified speakers instead of overwriting with `"unknown"`. User accepted the merged output for this session.

### enhance_audio on full 135-min files
- Initial run did not set `skip_enhance=True`, causing enhance to process full files (40+ min per file, GPU-bound).
- **Fix:** `skip_enhance=True` in `batch_process()` call. Enhance is irrelevant for speaker identification.

---

## Voice Library State (post-session)

```
transcript_samples/voices/
  Agustin/           # 6 segments (pre-existing)
  Patricio Renner/   # 5 segments (added this session from 20260320 recording)
  Ina Gonzalez/      # 5 segments (added this session from 20260318 recording)
  Cristobal/
  Francisco/
  Jolyon/
  Manuel/
  _unknown/          # empty (stale entries removed)
```

---

## Open Issues

1. **Multi-unknown speaker separation:** `core_analysis` must preserve diarization tags for unknown speakers. Currently all unknowns collapse to one tag, preventing per-speaker clip extraction.
2. **Threshold is empirical, not model-calibrated:** 0.40 was derived from one recording pair. Different microphone/environment combinations may require re-calibration. Consider exposing `threshold` as a parameter in `batch_process()`.
3. **Voice library quality:** Patricio Renner and Ina Gonzalez samples come from mobile recordings (lossy, variable SNR). Re-collecting from cleaner sources would improve future identification accuracy.
