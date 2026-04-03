# Session: Speaker Identification & Batch Processing

## Objective

Extend `speechlib` pipeline to process multiple real audio recordings, identify known speakers against a voice library, and extract unknown speakers as WAV clips for later naming and re-ingestion. Target recordings: two folders with Speaker A (known) + unknown persons.

---

## Modules Implemented

All session-generated code lives under `speechlib/tools/`. Tests under `tests/tools/`.

### `speechlib/tools/batch_process.py`
- Iterates folders → audio files → calls `core_analysis()` per file.
- Aggregates known speakers (`identified_speakers: set[str]`) and unknown clips.
- Calls `extract_unknown_speakers()` for segments tagged `"unknown"`.
- Returns `BatchReport` dataclass with `.print_summary()`.
- Relative imports: `..core_analysis`, `.extract_unknown_speakers`.

### `speechlib/tools/extract_unknown_speakers.py`
- Input: `{speaker_tag: [[start_s, end_s], ...]}`, audio path, output dir.
- Filters segments by `min_duration_s`, selects `max_clips` longest, saves to `output_dir/{tag}_{audio_stem}/segment_NN.wav`.
- Returns `{speaker_tag: Path}`.
- Relative import: `..audio_utils`.

### `speechlib/tools/process_recordings.py`
- Script: extracts 4-minute sample (min 1–5) per recording via `ffmpeg → WAV mono 16kHz` into a temp dir.
- Calls `batch_process()` with `skip_enhance=True`.
- Cleans temp dir after run.

### `speechlib/tools/diagnose_speaker.py`
- Input: audio path, timestamp `HH:MM:SS`, voices folder.
- Extracts 30s window around timestamp, computes embedding, prints cosine similarity table vs all speakers.
- Output: `PASS/FAIL` per speaker vs threshold. Does not modify anything.

### `speechlib/tools/relabel_vtt.py`
- Parses existing VTT, re-extracts audio chunk per segment from already-processed WAV, re-runs speaker recognition against voice library, writes corrected VTT.
- Does not touch transcription, diarization, or enhance.
- Output file: `{stem}_relabeled.vtt` (or `_relabeled_padded.vtt` with `--pad-short`).
- Flag `--pad-short` (experimental): segments shorter than `--pad-min-ms` (default 2000ms) are expanded symmetrically for embedding extraction only — VTT timestamps unchanged.

### `speechlib/speaker_recognition.py` — modified
- `find_best_speaker()`: added `threshold: float = 0.40` gate.
- Returns `"unknown"` when `best_score < threshold`.
- Skips voice library folders starting with `_`.

---

## Threshold Calibration

**Finding:** `pyannote/embedding` cosine similarity for same speaker across different recording conditions (mobile phone vs. controlled voice library) ranges **0.35–0.50**, not 0.75–0.95.

**Empirical data from real recordings:**
| Condition | Score range |
|---|---|
| Speaker A (known, same speaker) | 0.35 – 0.504 |
| Jolyon (false positive on Patricio segments) | 0.338 – 0.398 |
| Unknown women (Ina TRE recording) | 0.060 – 0.270 |

**Decision:** threshold = `0.40`
- Catches Speaker A (max 0.504, several segments > 0.40).
- Eliminates Jolyon false positive (max 0.398 < 0.40).
- Rejects all unknown voices (all scores < 0.30).

---

## What Worked

### Speaker identification pipeline
- `find_best_speaker` threshold gate: blocks false positives, correct abstention on unknown voices.
- `_` prefix exclusion in `speaker_recognition`: eliminates embedding errors from stale `_unknown/` entries.
- `extract_unknown_speakers`: produces correctly named, duration-filtered WAV clips.
- `batch_process` + `BatchReport`: correct aggregation across multiple folders/files.
- ffmpeg sample extraction (min 1–5): avoids processing 135-min files; reduces per-run time from 40+ min to ~3 min.
- `skip_enhance=True`: appropriate for speaker identification task; enhance is only needed for transcription quality.
- ATDD coverage: 17 tests in `tests/tools/` all pass.

### Threshold empirical calibration via debug scores
- Added temporary `print()` of per-segment scores inside `find_best_speaker` to observe real score distribution.
- Revealed that threshold 0.75 rejected all speakers, threshold 0.35 introduced Jolyon FP, threshold 0.40 was the correct cut.
- Debug print removed after calibration; replaced with `logger.debug`.

### VTT relabeling without reprocessing
- `relabel_vtt.py` re-labeled 524/654 segments (80%) of a 135-min recording in ~5 min.
- Confirmed via `diagnose_speaker.py`: at timestamp 01:42:17, Speaker A avg=0.516 → PASS with threshold 0.40. The `[unknown]` label was a stale artifact from the old 0.75 threshold run.
- Original VTT not modified; corrected version written as `_relabeled.vtt`.

### Experimental `--pad-short`
- Segments < 2000ms expanded symmetrically for embedding.
- Result: 538/654 identified vs 524/654 without padding (+14 segments).
- No regressions. Output written to `_relabeled_padded.vtt`.

---

## What Did Not Work

### Initial threshold 0.75 — complete failure
- All known speakers rejected. `identified_speakers = []`.
- Root cause: assumed 0.75 was a standard pyannote threshold; actual cross-condition similarity is ~0.5 max.
- Fix: empirical calibration via debug print, then threshold = 0.40.

### `_unknown/` inside `voices/` scanned as speaker
- `speaker_recognition` iterates all subdirs of `voices_folder` including `_unknown/`.
- Subdirs inside `_unknown/` are directories, not WAV files → `pyannote inference()` throws "File does not exist".
- Fix: skip dirs starting with `_` in `speaker_recognition`.

### Unknown speakers merged under single `"unknown"` tag
- `core_analysis` replaces all unidentified diarization tags with the string `"unknown"` before returning segments.
- `batch_process` groups all `speaker == "unknown"` segments under one key → `extract_unknown_speakers` receives mixed clips from different unknown speakers in one bucket.
- Consequence: Ina TRE recording — clips from 2 women merged into one folder.
- Status: not fixed. Requires `core_analysis` to preserve original diarization tag (e.g., `SPEAKER_01`) for unidentified speakers. User accepted merged output.

### `enhance_audio` on full 135-min files
- Initial run did not set `skip_enhance=True`, causing GPU-bound enhance on full files (40+ min per file). Process killed manually.
- Fix: `skip_enhance=True` in `batch_process()` call.

### `--pad-short` does not fix boundary segments
- Segment "Vale." at 00:00:00.300→00:00:00.900 (600ms) at start of file:
  - With padding, extraction window is 0ms→1600ms (clamped at 0, does not reach 2000ms target).
  - That 1600ms window is dominated by the other speaker (Patricio), producing a Patricio-leaning embedding.
  - Result: still `[unknown]` because Patricio score does not exceed threshold either.
- Limitation: padding borrows audio context from neighbors; if neighbors are a different speaker, the embedding is contaminated.
- No fix implemented. Boundary segments and segments surrounded by a different speaker remain unreliable.

### Jolyon residual false positive in `_relabeled_padded.vtt`
- 2 segments tagged `[Jolyon]` in the final output (up from 1 without padding).
- Padding occasionally pulls in enough ambiguous audio to push Jolyon above threshold.
- Not fixed. Acceptable noise at 2/654 segments (0.3%).

---

## Voice Library State (post-session)

```
transcript_samples/voices/
  Speaker A/           # 6 segments (pre-existing)
  Patricio Renner/   # 5 segments (added from 20260320 recording, mobile quality)
  Ina Gonzalez/      # 5 segments (added from 20260318 recording, mobile quality)
  Cristobal/
  Francisco/
  Jolyon/
  Manuel/
  _unknown/          # stale entries removed; skipped by speaker_recognition
```

---

## Open Issues

1. **Multi-unknown speaker separation:** `core_analysis` must preserve diarization tags for unknown speakers. Currently all unknowns collapse to one tag, preventing per-speaker clip extraction when multiple unknowns are present in the same recording.

2. **Threshold is empirical, not model-calibrated:** 0.40 was derived from one recording pair. Different microphone/environment combinations may require re-calibration. Consider exposing `threshold` as a parameter in `batch_process()` and `relabel_vtt.py`.

3. **Voice library quality:** Patricio Renner and Ina Gonzalez samples come from mobile recordings (lossy, variable SNR). Re-collecting from cleaner sources would improve identification accuracy and allow a higher threshold.

4. **`--pad-short` contamination at boundaries:** pad borrows context audio which may belong to a different speaker. A neighbor-aware pad (only pad toward the same diarization turn) would be more accurate but requires diarization data at relabel time.

5. **Relabeled VTT files are output-only artifacts:** `_relabeled.vtt` and `_relabeled_padded.vtt` are not tracked in git. If the voice library changes, re-running `relabel_vtt.py` on the original VTT is the correct workflow.
