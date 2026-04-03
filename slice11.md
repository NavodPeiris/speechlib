# Slice 11: Propagate threshold in speaker_recognition

## Problem
`detect_unknown_speakers` accepts `threshold` parameter but never passes it to `speaker_recognition`. `speaker_recognition` also doesn't accept `threshold`, so it always uses default 0.40 from `find_best_speaker`. Result: speakers with similarity 0.35-0.40 cannot be tuned to appear as unknown.

## Changes

### speechlib/speaker_recognition.py

1. **Line 99**: Added `threshold` parameter to `speaker_recognition`:
```python
def speaker_recognition(file_name, voices_folder, segments, wildcards,
                        threshold: float = SPEAKER_SIMILARITY_THRESHOLD):
```

2. **Line 150**: Pass `threshold` to `find_best_speaker`:
```python
best_speaker = find_best_speaker(test_emb, speaker_embeddings, threshold)
```

3. **Line 214**: Pass `threshold` in `detect_unknown_speakers`:
```python
name = speaker_recognition(str(audio_path), str(voices_folder), segments, [], threshold)
```

### tests/conftest.py

Added voice paths for tests:
```python
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
VOICES_DIR = EXAMPLES_DIR / "voices"
OBAMA_VOICES = VOICES_DIR / "obama"
ZACH_VOICES = VOICES_DIR / "zach"
```

### tests/test_acceptance_speaker_recognition_threshold.py

Created new file with parametrized tests:
- `test_threshold_behavior[0.4-speaker_a]`: threshold=0.40, sim=0.45 → recognized
- `test_threshold_behavior[0.5-unknown]`: threshold=0.50, sim=0.45 → unknown
- `test_threshold_propagation[0.4-True]`: detect threshold=0.40 → empty
- `test_threshold_propagation[0.5-False]`: detect threshold=0.50 → has segments

Uses real audio from examples/voices/obama/ but mocks embeddings for controlled similarity (avoids personal names in test code).

### tests/test_acceptance_detect_unknown_speakers.py

Updated 3 `fake_recognition` mocks to accept threshold parameter:
```python
def fake_recognition(file, voices, segments, wildcards, threshold=SPEAKER_SIMILARITY_THRESHOLD):
```

Added import: `from speechlib.speaker_recognition import SPEAKER_SIMILARITY_THRESHOLD`

## Verification
```bash
python -m pytest tests/test_acceptance_speaker_recognition_threshold.py -v  # 4 passed
python -m pytest -q -m "not e2e and not slow"  # 195 passed
```
