# Plan: Fix PreProcessor public API

**Context:** `PreProcessor` methods (`re_encode`, `convert_to_mono`, `convert_to_wav`)
pass raw strings to functions that now expect `AudioState`. Broken since Slice 2-4
of fix_audiostate.md.

**Methodology:** ATDD + TDD + One-Piece-Flow (devdocs/standard-atdd-tdd.md)

**Public contract to preserve:**
- Input: file path as `str`
- Output: file path as `str` (path of resulting file)
- Source file never modified

---

## Acceptance Test (RED until production is done)

```python
# tests/test_acceptance_preprocessor.py

def test_preprocessor_full_chain_preserves_source(tmp_path):
    """
    PreProcessor methods chain correctly and never modify the source.
    """
    source = make_wav(tmp_path / "source.wav", channels=2, sampwidth=1)
    original = source.read_bytes()

    prep = PreProcessor()
    wav_path  = prep.convert_to_wav(str(source))
    mono_path = prep.convert_to_mono(wav_path)
    enc_path  = prep.re_encode(mono_path)

    assert source.read_bytes() == original
    assert Path(enc_path).exists()
    assert Path(enc_path) != source
```

---

## Slice 5 — Fix PreProcessor

**Write AT** → RED.

**Write unit tests** `tests/test_preprocessor.py` → RED:

```python
def test_convert_to_wav_returns_str(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    prep = PreProcessor()
    result = prep.convert_to_wav(str(wav))
    assert isinstance(result, str)
    assert result.endswith(".wav")

def test_convert_to_mono_returns_str(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    prep = PreProcessor()
    result = prep.convert_to_mono(str(wav))
    assert isinstance(result, str)
    assert Path(result).exists()

def test_re_encode_returns_str(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    prep = PreProcessor()
    result = prep.re_encode(str(wav))
    assert isinstance(result, str)
    assert Path(result).exists()

def test_convert_to_wav_source_unchanged(tmp_path):
    wav = make_wav(tmp_path / "audio.wav")
    original = wav.read_bytes()
    PreProcessor().convert_to_wav(str(wav))
    assert wav.read_bytes() == original

def test_convert_to_mono_source_unchanged(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", channels=2)
    original = wav.read_bytes()
    PreProcessor().convert_to_mono(str(wav))
    assert wav.read_bytes() == original

def test_re_encode_source_unchanged(tmp_path):
    wav = make_wav(tmp_path / "audio.wav", sampwidth=1)
    original = wav.read_bytes()
    PreProcessor().re_encode(str(wav))
    assert wav.read_bytes() == original
```

**Production** — `speechlib/speechlib.py`, replace PreProcessor methods:

```python
def re_encode(self, file):
    from pathlib import Path
    state = AudioState(source_path=Path(file), working_path=Path(file))
    result = re_encode(state)
    return str(result.working_path)

def convert_to_mono(self, file):
    from pathlib import Path
    state = AudioState(source_path=Path(file), working_path=Path(file))
    result = convert_to_mono(state)
    return str(result.working_path)

def convert_to_wav(self, file):
    from pathlib import Path
    state = AudioState(source_path=Path(file), working_path=Path(file))
    result = convert_to_wav(state)
    return str(result.working_path)
```

Update `examples/preprocess.py` to capture return values:

```python
wav_file  = prep.convert_to_wav(file)
mono_file = prep.convert_to_mono(wav_file)
enc_file  = prep.re_encode(mono_file)
```

Run suite → AT GREEN, all unit tests GREEN. Commit + push.

---

## Files touched

| File | Change |
|---|---|
| `speechlib/speechlib.py` | PreProcessor methods wrap AudioState internally, return `str` |
| `tests/test_preprocessor.py` | new unit tests |
| `tests/test_acceptance_preprocessor.py` | new AT |
| `examples/preprocess.py` | capture return values to propagate path through chain |
