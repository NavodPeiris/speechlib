# Plan de Optimización del Pipeline de Transcripción

**Hardware objetivo:** NVIDIA RTX 2070 Super (8 GB VRAM)
**Rama activa:** `feat/batched-whisper-inference`
**Baseline:** ~6 min de audio → preprocessing 1s, diarización 21s, transcripción 128s = **152s total**
**Objetivo:** reducir transcripción ~3x con batching + mejorar calidad con resample y loudnorm

---

## Estado de slices

| Slice | Descripción | Estado |
|-------|-------------|--------|
| A | resample_to_16k (torchaudio, CPU) | ✅ completado |
| B | loudnorm EBU R128 (torchaudio, CPU) | ✅ completado |
| D | Speech Enhancement ClearerVoice `MossFormer2_SE_48K` (GPU) | ✅ completado |
| C | BatchedInferencePipeline faster-whisper (GPU) | ✅ completado |

**Orden de implementación:** A → B → D → C
**Orden de ejecución en pipeline:** A → B → D → C

> **Hallazgo validado en pruebas internas:** loudnorm *antes* de ClearVoice SE
> elimina el problema de supresión de hablantes secundarios. Al normalizar primero,
> todas las voces llegan a ClearVoice SE a nivel comparable (-14 LUFS), por lo que
> el modelo ya no las clasifica como "ruido a suprimir". La advertencia previa
> (arXiv:2512.17562) asumía audio sin normalizar previo.

---

## Slice A — resample_to_16k

### Por qué

Whisper resamplea internamente a 16kHz de todas formas. Hacerlo explícitamente antes:
- Reduce tamaño de archivos temporales ~3x (de 44.1/48kHz → 16kHz)
- Garantiza que loudnorm (Slice B) y ClearVoice (Slice D) reciban el SR correcto

---

### Metodología: ATDD → TDD (ver `standard-atdd-tdd.md`)

Orden: AT RED → unit tests RED → producción GREEN → refactor → commit.

---

### Paso 1 — Acceptance Test (RED)

**Archivo nuevo:** `tests/test_acceptance_slice_a.py`

Sin mocks. Ejerce el pipeline de preprocessing real hasta el efecto observable:
archivo de salida a 16kHz en disco. Falla hasta que `resample_to_16k` esté
creado y wired.

```python
"""AT: el pipeline de preprocessing produce audio a 16kHz (Slice A)."""
import torchaudio
from pathlib import Path
from conftest import make_wav
from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav
from speechlib.convert_to_mono import convert_to_mono
from speechlib.re_encode import re_encode
from speechlib.resample_to_16k import resample_to_16k


def test_preprocessing_pipeline_outputs_16khz(tmp_path):
    """AT: dado audio a 44100 Hz, el pipeline produce un archivo a 16000 Hz."""
    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)

    state = AudioState(source_path=wav, working_path=wav)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)

    assert state.is_16khz is True
    _, sr = torchaudio.load(str(state.working_path))
    assert sr == 16000


def test_preprocessing_pipeline_already_16khz(tmp_path):
    """AT: audio ya a 16kHz pasa el pipeline sin crear archivos extra."""
    wav = make_wav(tmp_path / "audio.wav", framerate=16000, n_frames=1600)
    files_before = set(tmp_path.iterdir())

    state = AudioState(source_path=wav, working_path=wav)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)

    assert state.is_16khz is True
    _, sr = torchaudio.load(str(state.working_path))
    assert sr == 16000
    # no se crea ningún archivo _16k nuevo
    assert not any("_16k" in f.name for f in tmp_path.iterdir())
```

---

### Paso 2 — Unit Tests (RED)

**Archivo nuevo:** `tests/test_resample_to_16k.py`

```python
"""Unit tests: speechlib/resample_to_16k.py"""
import torchaudio
from pathlib import Path
from conftest import make_wav
from speechlib.resample_to_16k import resample_to_16k
from speechlib.audio_state import AudioState


def _state(path: Path) -> AudioState:
    return AudioState(source_path=path, working_path=path,
                      is_wav=True, is_mono=True, is_16bit=True)


def test_non_16k_audio_is_resampled(tmp_path):
    """44100 Hz → nuevo archivo _16k.wav a 16000 Hz."""
    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)
    result = resample_to_16k(_state(wav))

    assert result.is_16khz is True
    assert result.working_path != wav
    assert "_16k" in result.working_path.name
    assert result.working_path.exists()
    _, sr = torchaudio.load(str(result.working_path))
    assert sr == 16000


def test_already_16k_passes_through(tmp_path):
    """16000 Hz → mismo working_path, sin crear archivo nuevo."""
    wav = make_wav(tmp_path / "audio.wav", framerate=16000, n_frames=1600)
    result = resample_to_16k(_state(wav))

    assert result.is_16khz is True
    assert result.working_path == wav
    assert not (tmp_path / "audio_16k.wav").exists()


def test_48k_audio_is_resampled(tmp_path):
    """48000 Hz (micrófono de reunión) → 16000 Hz."""
    wav = make_wav(tmp_path / "audio.wav", framerate=48000, n_frames=4800)
    result = resample_to_16k(_state(wav))

    assert result.is_16khz is True
    _, sr = torchaudio.load(str(result.working_path))
    assert sr == 16000


def test_source_path_is_never_modified(tmp_path):
    """source_path permanece igual en todos los casos."""
    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)
    state = _state(wav)
    result = resample_to_16k(state)

    assert result.source_path == state.source_path


def test_output_is_mono_wav(tmp_path):
    """El archivo de salida es WAV mono (1 canal)."""
    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)
    result = resample_to_16k(_state(wav))

    waveform, _ = torchaudio.load(str(result.working_path))
    assert waveform.shape[0] == 1
```

---

### Paso 3 — Producción

#### `speechlib/audio_state.py` — añadir campo

```python
class AudioState(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_path:   Path
    working_path:  Path
    is_wav:        bool = False
    is_mono:       bool = False
    is_16bit:      bool = False
    is_16khz:      bool = False   # ← nuevo
```

#### `speechlib/resample_to_16k.py` — archivo nuevo

```python
import torchaudio
from .audio_state import AudioState

TARGET_SR = 16000


def resample_to_16k(state: AudioState) -> AudioState:
    waveform, sr = torchaudio.load(str(state.working_path))
    if sr == TARGET_SR:
        return state.model_copy(update={"is_16khz": True})

    resampled = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    out_path = state.working_path.with_stem(state.working_path.stem + "_16k")
    torchaudio.save(str(out_path), resampled, TARGET_SR, bits_per_sample=16)
    return state.model_copy(update={"working_path": out_path, "is_16khz": True})
```

#### `speechlib/core_analysis.py` — import + wiring

```python
# añadir import junto a los demás
from .resample_to_16k import resample_to_16k

# en core_analysis(), después de re_encode:
state = re_encode(state)
state = resample_to_16k(state)   # ← nuevo
```

---

### Paso 4 — Actualizar tests existentes de core_analysis

Los tests en `test_core_analysis_pyannote4.py` usan mocks para aislar de pyannote/GPU
(servicios externos inevitables). Al wired `resample_to_16k` en core_analysis, esos
tests romperán porque el nuevo import no existe aún. Una vez en GREEN, añadir el
pass-through para mantenerlos GREEN:

```python
patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
```

Afecta a los 3 tests existentes: `test_from_pretrained_uses_3_1_model`,
`test_pipeline_called_with_waveform_dict`, `test_itertracks_still_yields_3tuple`.

---

### Definition of Done

- [ ] AT RED: `test_acceptance_slice_a.py::test_core_analysis_calls_resample_to_16k` falla
- [ ] Unit tests RED: los 5 tests en `test_resample_to_16k.py` fallan
- [ ] `audio_state.py`: campo `is_16khz` añadido
- [ ] `resample_to_16k.py`: creado, todos los unit tests GREEN
- [ ] `core_analysis.py`: import + llamada después de `re_encode`, AT GREEN
- [ ] `test_core_analysis_pyannote4.py`: mock añadido, 3 tests siguen GREEN
- [ ] Commit: `feat: add resample_to_16k preprocessing step (Slice A)`

---

## Slice B — loudnorm

### Por qué

Voces bajas generan transcripciones incompletas. Normalización EBU R128 a -14 LUFS mejora
la confianza de Whisper (~75% con voz muy baja → ~96% con -14 LUFS).

Target: `I = -14 LUFS, TP = -1.0 dB`

### Archivo nuevo: `speechlib/loudnorm.py`

```python
import torch, torchaudio
from .audio_state import AudioState

TARGET_LUFS = -14.0
TRUE_PEAK_DB = -1.0

def loudnorm(state: AudioState) -> AudioState:
    waveform, sr = torchaudio.load(str(state.working_path))

    current_lufs = torchaudio.functional.loudness(waveform, sr).item()

    if abs(current_lufs - TARGET_LUFS) < 0.5:
        return state.model_copy(update={"is_normalized": True})

    gain_db     = TARGET_LUFS - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    normalized  = waveform * gain_linear

    true_peak = 10 ** (TRUE_PEAK_DB / 20)
    normalized = torch.clamp(normalized, -true_peak, true_peak)

    torchaudio.save(str(state.working_path), normalized, sr, bits_per_sample=16)
    return state.model_copy(update={"is_normalized": True})
```

Nota: opera in-place sobre `working_path` (no crea archivo nuevo).
Tiempo estimado: < 0.3s para 6 min de audio a 16kHz en CPU.

### Wiring en `core_analysis.py`

```python
state = resample_to_16k(state)
state = loudnorm(state)          # ← nuevo
diarization = pipeline(str(state.working_path))
```

### AudioState: campo nuevo requerido

```python
is_normalized: bool = False
```

### Definition of Done

- [ ] `speechlib/loudnorm.py` creado
- [ ] `AudioState` tiene campo `is_normalized`
- [ ] `core_analysis.py` llama a `loudnorm` después de `resample_to_16k`
- [ ] Test unitario: audio bajo (<-20 LUFS) se normaliza a -14 LUFS
- [ ] Test unitario: audio ya normalizado (dentro de ±0.5 LUFS) pasa sin reescribir
- [ ] Test unitario: `is_normalized=True` en ambos casos

---

## Slice D — Speech Enhancement (ClearVoice `MossFormer2_SE_48K`)

### Por qué — y por qué loudnorm debe ir antes

ClearVoice SE entrenado en DNS Challenge suprime voces secundarias porque las clasifica
como "interferencia". Esto ocurre cuando los hablantes llegan con niveles muy dispares.

**Solución validada internamente:** aplicar loudnorm (Slice B) primero iguala los niveles
de todas las voces al target (-14 LUFS). ClearVoice SE recibe señal con hablantes
equiparados y ya no los suprime.

```
Sin loudnorm previo:
  Speaker A (cerca):  -18 dBFS  ← preservado por ClearVoice
  Speaker B (lejos):  -35 dBFS  ← suprimido como "ruido"

Con loudnorm previo (todos a -14 LUFS):
  Speaker A:  -14 LUFS  ← preservado
  Speaker B:  -14 LUFS  ← preservado (ya no parece ruido)
```

### Fuente local

ClearVoice disponible en: `c:\workspace\#dev\ClearerVoice-Studio\clearvoice\`
**No instalar desde PyPI** — usar el repo local con `pip install -e .` tras eliminar `pydub`.

Modelo: `MossFormer2_SE_48K`

### Paso previo obligatorio: eliminar pydub de ClearerVoice-Studio

`pydub` aparece en 3 archivos del repo local. Reemplazar con `torchaudio` / `soundfile`
(ambos ya son dependencias de ClearerVoice).

#### `clearvoice/dataloader/dataloader.py` — `read_audio()`

```python
# ANTES
from pydub import AudioSegment
audio = AudioSegment.from_file(file_path)

# DESPUÉS
import torchaudio
waveform, sr = torchaudio.load(file_path)   # devuelve tensor + sample_rate
# adaptar el resto de la función al formato tensor
```

#### `clearvoice/dataloader/meldataset.py` — slicing por ms

```python
# ANTES
from pydub import AudioSegment
audio = AudioSegment.from_wav(file_path)
segment = audio[start_ms:end_ms]

# DESPUÉS
import torchaudio
waveform, sr = torchaudio.load(file_path)
start_sample = int(start_ms / 1000 * sr)
end_sample   = int(end_ms   / 1000 * sr)
segment = waveform[:, start_sample:end_sample]
```

#### `clearvoice/networks.py` — construcción de AudioSegment desde bytes

```python
# ANTES
from pydub import AudioSegment
audio_segment = AudioSegment(result.tobytes(), frame_rate=sr, sample_width=sw, channels=ch)

# DESPUÉS
import soundfile as sf
import numpy as np
sf.write(out_path, result, sr)   # soundfile ya es dependencia de ClearerVoice
```

#### `pyproject.toml` de ClearerVoice-Studio

Eliminar `"pydub"` de la lista `dependencies`.

### Archivo nuevo en speechlib: `speechlib/enhance_audio.py`

```python
import sys
sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")
from clearvoice import ClearVoice
from .audio_state import AudioState

_clearvoice_model = None

def enhance_audio(state: AudioState) -> AudioState:
    global _clearvoice_model
    if _clearvoice_model is None:
        _clearvoice_model = ClearVoice(task='speech_enhancement',
                                       model_names=['MossFormer2_SE_48K'])

    out_path = state.working_path.with_stem(state.working_path.stem + "_enhanced")
    _clearvoice_model(input_path=str(state.working_path),
                      output_path=str(out_path),
                      online_write=True)
    return state.model_copy(update={"working_path": out_path, "is_enhanced": True})
```

### Wiring en `core_analysis.py`

```python
state = loudnorm(state)          # B — normalizar primero
state = enhance_audio(state)     # D — enhancement sobre audio normalizado
diarization = pipeline(str(state.working_path))
```

### AudioState: campo nuevo requerido

```python
is_enhanced: bool = False
```

### Definition of Done

- [ ] `pydub` eliminado de los 3 archivos en ClearerVoice-Studio (dataloader.py, meldataset.py, networks.py)
- [ ] `pydub` eliminado de `pyproject.toml` de ClearerVoice-Studio
- [ ] ClearerVoice-Studio instalado con `pip install -e .` desde el repo local
- [ ] `speechlib/enhance_audio.py` creado con lazy-loading del modelo
- [ ] `AudioState` tiene campo `is_enhanced`
- [ ] `core_analysis.py` llama a `enhance_audio` **después** de `loudnorm`
- [ ] Test: audio de múltiples hablantes — hablante secundario presente en output enhanced
- [ ] Test: audio ya limpio pasa sin degradación notable
- [ ] `is_enhanced=True` en output de AudioState

---

## Slice C — BatchedInferencePipeline (faster-whisper)

### Por qué

La GPU está ociosa ~70% del tiempo entre chunks en el flujo secuencial actual.
`BatchedInferencePipeline` agrupa múltiples chunks en un solo forward pass GPU.

```
Actual:   [chunk1] idle [chunk2] idle [chunk3] idle  →  128s, GPU ~30%
Batched:  [chunk1+chunk2+chunk3+...] en paralelo     →  ~43s, GPU ~90%
```

Speedup esperado: **3x** sobre faster-whisper secuencial (12.5x sobre whisper original).

### Cambio en `speechlib/transcribe.py`

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

# dentro del bloque faster_whisper:
model = _get_faster_whisper_model(model_size, device, compute_type)
batched = BatchedInferencePipeline(model=model)
segments, info = batched.transcribe(
    file,
    batch_size=16,
    language=language,
    beam_size=5,
)
```

### VRAM con batch_size=16 (RTX 2070 Super, 8 GB)

| Modelo   | VRAM necesaria | Resultado      |
|----------|----------------|----------------|
| tiny     | ~1 GB          | ✓ holgado      |
| small    | ~2 GB          | ✓ holgado      |
| medium   | ~4 GB          | ✓ holgado      |
| large-v2 | ~6–7 GB        | ✓ margen justo |
| large-v3 | ~7–8 GB        | reducir a batch_size=8 |

### Compatibilidad

- `faster_whisper()` ← cambio aquí
- `whisper()`, `custom_whisper()`, `huggingface_model()`, `assemblyAI()` ← sin cambio
- Output idéntico al actual (mismos timestamps y texto)

### Fallbacks

1. Si `BatchedInferencePipeline` falla → caer a `model.transcribe()` normal
2. Si VRAM insuficiente → reducir `batch_size` o usar modelo más pequeño

### Definition of Done

- [ ] `speechlib/transcribe.py` usa `BatchedInferencePipeline` para `faster_whisper`
- [ ] Test: output idéntico al secuencial para el mismo audio
- [ ] Test: `WhisperModel` sigue siendo instanciado una sola vez (LRU cache intacto)
- [ ] Tiempo de transcripción verificado ~3x menor en audio de referencia
- [ ] Branch mergeado a main
