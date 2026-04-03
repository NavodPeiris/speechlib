# Plan de Optimización del Pipeline de Transcripción

**Hardware objetivo:** NVIDIA RTX 2070 Super (8 GB VRAM)
**Rama:** `feat/batched-whisper-inference`
**Baseline:** ~6 min de audio → transcripción 128s (secuencial)
**Resultado medido:** transcripción 2.58s — **5.12x speedup** sobre baseline secuencial

---

## Estado de slices

| Slice | Descripción | Estado |
|-------|-------------|--------|
| A | resample_to_16k (torchaudio, CPU) | ✅ completado |
| B | loudnorm EBU R128 (torchaudio, CPU) | ✅ completado |
| D | Speech Enhancement ClearerVoice `MossFormer2_SE_48K` (GPU) | ✅ completado |
| C | BatchedInferencePipeline faster-whisper (GPU) | ✅ completado |

**Orden de ejecución en pipeline:** A → B → D → C

---

## Decisiones de diseño clave

### Slice D: loudnorm antes de ClearVoice SE (validado internamente)

ClearVoice SE entrenado en DNS Challenge suprime voces secundarias cuando los hablantes
llegan con niveles muy dispares (speaker lejano clasifica como "ruido").

**Solución:** aplicar `loudnorm` (Slice B) primero iguala todas las voces a -14 LUFS
antes de que ClearVoice las procese. Los tests reales con múltiples hablantes confirman
que la supresión desaparece con este orden.

```
Sin loudnorm previo:  Speaker A -18 dBFS (preservado) / Speaker B -35 dBFS (suprimido)
Con loudnorm previo:  Speaker A -14 LUFS (preservado) / Speaker B -14 LUFS (preservado)
```

### ClearVoice — fuente local

ClearVoice disponible en: `c:\workspace\#dev\ClearerVoice-Studio\clearvoice\`
Instalado con `pip install -e .` tras eliminar dependencia `pydub` (reemplazada con `soundfile`).

---

## Resumen de implementación

### Slice A — `speechlib/resample_to_16k.py`
- `resample_to_16k(state)` → `torchaudio.functional.resample` al SR objetivo 16000
- Pass-through si ya está a 16 kHz (no crea archivo nuevo)
- Tests: `test_resample_to_16k.py` (5 unit), `test_acceptance_slice_a.py` (2 AT)

### Slice B — `speechlib/loudnorm.py`
- `loudnorm(state)` → normalization EBU R128 a -14 LUFS, clamp true peak -1 dBTP
- In-place sobre `working_path`; skip si LUFS < -70 (silencio) o ya dentro de ±0.5 LUFS
- Tests: `test_loudnorm.py` (5 unit), `test_acceptance_slice_b.py` (2 AT)

### Slice D — `speechlib/enhance_audio.py`
- `enhance_audio(state)` → ClearVoice `MossFormer2_SE_48K` con lazy-loading global
- Output en `*_enhanced_out/MossFormer2_SE_48K/{original_filename}`
- Tests: `test_enhance_audio.py` (4 unit, `pytest.mark.slow`), `test_acceptance_slice_d.py` (2 AT)

### Slice C — `speechlib/transcribe.py`
- `BatchedInferencePipeline(model=model).transcribe(..., batch_size=16)`
- LRU cache (`@lru_cache(maxsize=4)`) en `_get_faster_whisper_model` preservado
- Tests: `test_acceptance_slice_c.py` (4 AT), `test_transcribe_cache.py` (4 unit)

---

## Benchmark medido

```
Audio: obama_zach_16k.wav (~6.6 min), model=base, RTX 2070 Super, CUDA float16

Sequential model.transcribe():      13.23s  (1081 palabras)
BatchedInferencePipeline(bs=16):     2.58s  (1057 palabras)
Speedup: 5.12x
Word overlap: ~98%
```

Reproducible con: `python benchmark_slice_c.py`
