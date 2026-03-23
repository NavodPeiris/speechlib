# Plan: Compresión opcional de audio post-enhance

## Contexto

Todas las conversiones del preprocessing (convert_to_wav, convert_to_mono, re_encode, resample_to_16k, loudnorm) usan **CPU** exclusivamente (torchaudio, wave, numpy, struct). Solo `enhance_audio` usa GPU.

Tras `enhance_audio`, el pipeline produce un WAV a 48kHz (~27 MB/min) que persiste en disco permanentemente. No existe paso de compresión para archival. La brecha: falta un módulo opcional que produzca una copia comprimida AAC para almacenamiento eficiente.

**Parámetros confirmados:** AAC mono 96kbps 16kHz loudnorm EBU R128 (-14 LUFS, -1 dBTP, LRA 9)

---

## Migración torchaudio → torchcodec

### Estado actual del ecosistema (verificado 2026-03-23)

| Paquete | Versión instalada | Estado |
|---------|-------------------|--------|
| torch | 2.10.0+cu126 | estable |
| torchaudio | 2.10.0+cu126 | **I/O deprecated** — `load/save` → torchcodec |
| torchcodec | 0.10.0 | **AudioEncoder disponible** — soporta AAC via .m4a |

### Timeline de deprecation de torchaudio

- **torchaudio 2.8** (agosto 2025): deprecation warnings en `torchaudio.load()` / `torchaudio.save()`
- **torchaudio 2.9** (fin 2025): remoción de APIs I/O
- **torchaudio 2.10** (actual): puente `load_with_torchcodec()` / `save_with_torchcodec()`
- **APIs que se MANTIENEN**: `torchaudio.functional.resample`, `torchaudio.functional.loudness`, `transforms`, `models`, `pipelines`

### Qué se depreca vs qué se mantiene

| API | Estado | Reemplazo |
|-----|--------|-----------|
| `torchaudio.load()` | DEPRECATED | `torchcodec.decoders.AudioDecoder()` |
| `torchaudio.save()` | DEPRECATED | `torchcodec.encoders.AudioEncoder.to_file()` |
| `torchaudio.functional.resample()` | **SE MANTIENE** | — |
| `torchaudio.functional.loudness()` | **SE MANTIENE** | — |

### torchcodec AudioEncoder — verificado experimentalmente

```python
from torchcodec.encoders import AudioEncoder
import torch

# samples: (1, N) float tensor en [-1, 1]
encoder = AudioEncoder(samples, sample_rate=48000)
encoder.to_file("output.m4a", bit_rate=96000, num_channels=1, sample_rate=16000)
```

- **Formatos verificados OK**: mp4, adts, mp3, wav, flac, ogg
- **Formato m4a**: funciona via `to_file("x.m4a")` pero NO via `to_tensor(format="m4a")`
- **Encoding es CPU-only**: no existe aceleración GPU para audio encoding (confirmado en torchcodec issue #164)
- **Loudnorm**: NO incluido en torchcodec — se aplica antes con `torchaudio.functional.loudness` + `torch.clamp` (que se mantiene)

### Impacto en el pipeline actual

Módulos speechlib que usan torchaudio I/O (deben migrar eventualmente):

| Módulo | Uso de torchaudio | Migración |
|--------|-------------------|-----------|
| `convert_to_wav.py` | `torchaudio.load()` / `torchaudio.save()` | → `AudioDecoder` / `AudioEncoder.to_file()` |
| `resample_to_16k.py` | `torchaudio.load()` / `torchaudio.save()` + `.functional.resample()` | I/O → torchcodec, resample se mantiene |
| `loudnorm.py` | `torchaudio.load()` / `torchaudio.save()` + `.functional.loudness()` | I/O → torchcodec, loudness se mantiene |
| `core_analysis.py` | `torchaudio.load()` para diarización | → `AudioDecoder` |

---

## Decisiones de diseño para compress_audio

1. **Usar torchcodec.encoders.AudioEncoder** (no FFmpeg subprocess) — ya instalado, API nativa Python/PyTorch, sin dependencia externa adicional
2. **No modifica AudioState** — la compresión produce un archivo de archival separado
3. **Comprime `state.working_path`** (el WAV enhanced/normalizado a 48kHz)
4. **Loudnorm ya aplicado** — el audio ya pasó por `loudnorm.py` antes de enhance; no se re-aplica
5. **Output junto al source**: `state.source_path.with_suffix(".m4a")`
6. **Lectura con torchcodec AudioDecoder** — consistente con la migración; no usar torchaudio.load deprecated
7. **Graceful degradation** — si torchcodec no está instalado, fallback a FFmpeg subprocess; si ninguno disponible, log warning

### Flujo de compress_audio

```
state.working_path (WAV 48kHz enhanced+normalized)
    ↓
AudioDecoder → torch.Tensor (1, N) float32
    ↓
AudioEncoder(samples, sample_rate=48000)
    ↓
.to_file("output.m4a", bit_rate=96000, num_channels=1, sample_rate=16000)
    ↓
state.source_path.with_suffix(".m4a") — archival AAC mono 96kbps 16kHz
```

### Optimización en el pipeline completo

```
                    Pipeline con compresión
                    =======================

preprocessing (CPU)  →  enhance (GPU)  →  compress (CPU, torchcodec)
                                           ↕ paralelo con:
                                          diarization (GPU)  →  transcription (GPU)
                                                                    ↓
                                                               write_vtt
```

La compresión es CPU-only y puede ejecutarse en paralelo con diarización/transcripción (que usan GPU). Para la primera implementación será sincrónico; la paralelización es una optimización futura.

---

## Secuencia ATDD

### Slice 1: AT RED — `tests/test_acceptance_compress_audio.py`

E2E con audio real (`examples/obama_zach.wav`), sin mocks. Patrón de `test_e2e_wx3_advantages.py`.

| Test | Verifica |
|------|----------|
| `test_compressed_file_exists` | Existe `.m4a` junto al source tras `compress=True` |
| `test_compressed_is_aac_mono_16k` | ffprobe: codec=aac, channels=1, sample_rate=16000 |
| `test_compressed_bitrate_near_96k` | ffprobe: bit_rate entre 80k-112k |
| `test_no_compressed_file_by_default` | Sin `compress=True`, no se produce `.m4a` |
| `test_pipeline_output_unchanged` | Segmentos válidos con compresión activa |

Fixture de sesión: `core_analysis(..., compress=True, skip_enhance=True, modelSize="base")` para velocidad.

### Slice 2: Implementar `speechlib/compress_audio.py`

```python
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from pathlib import Path
from .step_timer import timed

@timed("compress_audio")
def compress_audio(source: Path, output: Path) -> Path | None:
    """Produce archival AAC copy: mono 96kbps 16kHz."""
    decoder = AudioDecoder(str(source))
    result = decoder.get_all_samples()
    encoder = AudioEncoder(result.data, sample_rate=result.sample_rate)
    encoder.to_file(str(output), bit_rate=96000, num_channels=1, sample_rate=16000)
    return output
```

### Slice 3: Wire en `core_analysis.py`

- Nuevo parámetro: `compress: bool = False`
- Import: `from .compress_audio import compress_audio`
- Después de enhance (línea ~76):
  ```python
  if compress:
      compress_audio(state.working_path, state.source_path.with_suffix(".m4a"))
  ```

### Slice 4: Verificar GREEN

---

## Archivos

| Archivo | Acción |
|------|--------|
| `tests/test_acceptance_compress_audio.py` | CREAR — AT RED E2E |
| `speechlib/compress_audio.py` | CREAR — módulo torchcodec |
| `speechlib/core_analysis.py` | MODIFICAR — añadir param `compress`, llamar `compress_audio` |

## Archivos de referencia (no modificar)

- `speechlib/loudnorm.py` — patrón de módulo con `@timed`
- `speechlib/enhance_audio.py` — confirma working_path post-enhance (48kHz)
- `speechlib/audio_state.py` — sin cambios (no agregar `is_compressed`)
- `speechlib/step_timer.py` — `@timed` decorator API

## Migración futura de torchaudio I/O

Los siguientes módulos deberán migrar de `torchaudio.load/save` a `torchcodec.AudioDecoder/AudioEncoder` antes de torchaudio 2.9:

- `speechlib/convert_to_wav.py`
- `speechlib/resample_to_16k.py`
- `speechlib/loudnorm.py`
- `speechlib/core_analysis.py` (línea 83)

`compress_audio.py` será el primer módulo que use torchcodec, estableciendo el patrón para la migración.

## Verificación

```bash
# AT RED
pytest tests/test_acceptance_compress_audio.py -v -s -m e2e

# Implementar → AT GREEN
pytest tests/test_acceptance_compress_audio.py -v -s -m e2e

# Suite completa
pytest -q
```
