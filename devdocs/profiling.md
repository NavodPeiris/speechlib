# Plan: Instrumentación del Pipeline para Profiling

## Objetivo

Medir el tiempo real de cada paso del pipeline y el uso de VRAM para identificar
cuellos de botella y optimizar velocidad en CPU y GPU.

---

## Estrategia: activación por variable de entorno

```bash
SPEECHLIB_PROFILE=1 python process_audio.py
```

Sin la variable, cero overhead en producción. Con ella, reporte completo al terminar.

---

## Módulo a crear: `speechlib/pipeline_profiler.py`

### Decorador `@timed` — wrappea cualquier función sin modificarla

```python
import time
import torch
from functools import wraps

_GPU_STEPS = {"enhance_audio", "diarization", "transcription"}

def timed(step_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _profiling_enabled():
                return func(*args, **kwargs)

            is_gpu = step_name in _GPU_STEPS and torch.cuda.is_available()

            if is_gpu:
                torch.cuda.synchronize()   # vaciar cola GPU antes de medir
            mem_before = _vram_mb() if is_gpu else None

            t0 = time.perf_counter()
            result = func(*args, **kwargs)

            if is_gpu:
                torch.cuda.synchronize()   # esperar que GPU termine antes de leer tiempo
            elapsed = time.perf_counter() - t0

            mem_after = _vram_mb() if is_gpu else None
            _record(step_name, elapsed, mem_before, mem_after)
            return result
        return wrapper
    return decorator
```

### Context manager `measure(name)` — para bloques inline

```python
from contextlib import contextmanager

@contextmanager
def measure(step_name: str, gpu: bool = False):
    if not _profiling_enabled():
        yield
        return

    if gpu and torch.cuda.is_available():
        torch.cuda.synchronize()
    mem_before = _vram_mb() if gpu else None
    t0 = time.perf_counter()

    yield

    if gpu and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mem_after = _vram_mb() if gpu else None
    _record(step_name, elapsed, mem_before, mem_after)
```

### Reporte al finalizar

```
=== Pipeline Profiling Report ===

Step                    Time      VRAM before  VRAM after   VRAM delta
----------------------  --------  -----------  -----------  ----------
convert_to_wav           0.312s        —            —            —
convert_to_mono          0.041s        —            —            —
re_encode                0.008s        —            —            —
resample_to_16k          0.093s        —            —            —
loudnorm                 0.187s        —            —            —
enhance_audio           18.432s    512 MB      2341 MB     +1829 MB
diarization             21.105s   2341 MB      3102 MB      +761 MB
transcription            2.580s   3102 MB      3240 MB      +138 MB
write_log_file           0.003s        —            —            —
─────────────────────────────────────────────────────────────────────
TOTAL                   42.761s
```

---

## Wiring en core_analysis.py

Aplicar `@timed` a cada función del pipeline:

```python
from .pipeline_profiler import timed, print_report

# decorar en el módulo de cada step:
@timed("resample_to_16k")
def resample_to_16k(state): ...

@timed("loudnorm")
def loudnorm(state): ...

@timed("enhance_audio")          # GPU — usa torch.cuda.synchronize
def enhance_audio(state): ...
```

Para los bloques inline en core_analysis (diarización, transcripción):

```python
with measure("diarization", gpu=True):
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

with measure("transcription", gpu=True):
    segment_out = wav_file_segmentation(...)
```

Llamar `print_report()` al final de `core_analysis()`.

---

## Por qué `torch.cuda.synchronize()` es obligatorio

PyTorch lanza operaciones GPU de forma **asíncrona**: la CPU encola el trabajo
y retorna inmediatamente. Sin `synchronize()`, `perf_counter()` mide solo el
tiempo de encolamiento (~microsegundos), no la ejecución real (~segundos).

```
Sin synchronize:   enhance_audio → 0.003s  ← FALSO (solo lanzamiento)
Con synchronize:   enhance_audio → 18.4s   ← REAL (ejecución GPU completa)
```

---

## Nivel 2 (opcional): análisis profundo con torch.profiler

Cuando el Nivel 1 identifica qué paso es el cuello de botella, activar:

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    with record_function("enhance_audio"):
        state = enhance_audio(state)

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("enhance_audio_trace.json")  # visualizar en chrome://tracing
```

---

## Nivel 3 (opcional): scalene — sin modificar código

```bash
pip install scalene
scalene --gpu speechlib/core_analysis.py
```

Produce análisis línea por línea con separación CPU / GPU / memoria.
Overhead: ~15% (aceptable para debugging, no para producción).

---

## Archivos a crear/modificar

| Acción | Archivo |
|---|---|
| Crear | `speechlib/pipeline_profiler.py` |
| Modificar | `speechlib/resample_to_16k.py` — añadir `@timed` |
| Modificar | `speechlib/loudnorm.py` — añadir `@timed` |
| Modificar | `speechlib/enhance_audio.py` — añadir `@timed` |
| Modificar | `speechlib/core_analysis.py` — añadir `measure()` para diarización y transcripción, llamar `print_report()` |
