# Plan: Compresión opcional de audio post-enhance

## Contexto

Todas las conversiones del preprocessing (convert_to_wav, convert_to_mono, re_encode, resample_to_16k, loudnorm) usan **CPU** exclusivamente (torchaudio, wave, numpy, struct). Solo `enhance_audio` usa GPU.

Tras `enhance_audio`, el pipeline produce un WAV a 48kHz (~27 MB/min) que persiste en disco permanentemente. No existe paso de compresión para archival. La brecha: falta un módulo opcional que produzca una copia comprimida AAC para almacenamiento eficiente.

**Parámetros confirmados:** AAC mono 96kbps 16kHz loudnorm EBU R128 (-14 LUFS, -1 dBTP, LRA 9)

---

## Decisiones de diseño

1. **No modifica AudioState** — la compresión produce un archivo de archival separado, no alimenta al pipeline (diarization/transcription siguen usando el WAV)
2. **Comprime `state.working_path`** (el WAV enhanced/normalizado) — mejor calidad que el source original
3. **Output junto al source**: `state.source_path.with_suffix(".m4a")` — el archival vive junto al input original
4. **FFmpeg via subprocess** — no existe uso previo de FFmpeg en el codebase; es la herramienta estándar para AAC
5. **Graceful degradation** — si FFmpeg no está instalado, log warning y continúa sin error
6. **Sincrónico** — FFmpeg en un audio de 5min toma <2s en CPU; no justifica threading

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
# Firma
def compress_audio(source: Path, output: Path) -> Path | None
```

- `shutil.which("ffmpeg")` check
- `subprocess.run` con: `-ac 1 -ar 16000 -c:a aac -b:a 96k -af "loudnorm=I=-14:TP=-1:LRA=9"`
- Decorado con `@timed("compress_audio")`
- Retorna Path o None si falla

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
| `speechlib/compress_audio.py` | CREAR — módulo FFmpeg |
| `speechlib/core_analysis.py` | MODIFICAR — añadir param `compress`, llamar `compress_audio` |

## Archivos de referencia (no modificar)

- `speechlib/loudnorm.py` — patrón de módulo con `@timed`
- `speechlib/enhance_audio.py` — confirma working_path post-enhance
- `speechlib/audio_state.py` — sin cambios (no agregar `is_compressed`)
- `speechlib/step_timer.py` — `@timed` decorator API

## Verificación

```bash
# AT RED
pytest tests/test_acceptance_compress_audio.py -v -s -m e2e

# Implementar → AT GREEN
pytest tests/test_acceptance_compress_audio.py -v -s -m e2e

# Suite completa
pytest -q
```
