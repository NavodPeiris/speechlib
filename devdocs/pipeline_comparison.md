# Análisis Visual: Pipeline Actual vs. BatchedInferencePipeline

> **Scope:** Pipeline completo implementado — Slices A, B, D, C.
> Pipeline de ejecución: `resample_to_16k → loudnorm → enhance_audio → diarize → transcribe (batched)`
> ClearVoice SE fue descartado inicialmente, pero luego reintegrado tras validar que
> `loudnorm`-first elimina la supresión de hablantes secundarios.

## Pipeline Actual (secuencial)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRANSCRIPCION ACTUAL (128 seg)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AUDIO (6 min)                                                              │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │  CHUNK 1    │    │  CHUNK 2    │    │  CHUNK N    │                   │
│  │  (0-30 seg) │    │ (30-60 seg) │    │(150-180 seg)│                   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│    ┌─────────┐         ┌─────────┐         ┌─────────┐                     │
│    │ GPU     │         │ GPU     │         │ GPU     │                     │
│    │ WAIT 1  │         │ WAIT 2  │         │ WAIT N  │                     │
│    └────┬────┘         └────┬────┘         └────┬────┘                     │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│    ┌─────────┐         ┌─────────┐         ┌─────────┐                     │
│    │RESULT 1 │         │RESULT 2 │         │RESULT N │                     │
│    └─────────┘         └─────────┘         └─────────┘                     │
│         │                   │                   │                           │
│         └───────────────────┴───────────────────┘                           │
│                             │                                               │
│                             ▼                                               │
│                    ┌────────────────┐                                       │
│                    │  CONCATENAR    │                                       │
│                    │   RESULTADOS   │                                       │
│                    └────────────────┘                                       │
│                                                                             │
│  TIEMPO:  CHUNK 1 + CHUNK 2 + ... + CHUNK N = SECUENCIAL                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


GPU Utilization (tiempo):
██████████████████████████████  (brecha = espera entre chunks)
██████████████████████████████
██████████████████████████████
         ▲
         │
    GPU espera que
    termine cada chunk
```

---

## Pipeline Optimizado (BatchedInferencePipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSCRIPCION BATCHED (~43 seg)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AUDIO (6 min)                                                              │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │  CHUNK 1    │    │  CHUNK 2    │    │  CHUNK N    │                   │
│  │  (0-30 seg) │    │ (30-60 seg) │    │(150-180 seg)│                   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                   │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             │                                               │
│                             ▼                                               │
│                    ┌────────────────┐                                       │
│                    │   BATCH QUEUE │  (recolectar chunks)                  │
│                    └────────┬───────┘                                       │
│                             │                                               │
│                             ▼                                               │
│                 ┌───────────────────────┐                                  │
│                 │    GPU PROCESA        │                                  │
│                 │  ┌─────┬─────┬─────┐ │                                  │
│                 │  │C1   │C2   │C3...│ │  TODOS EN PARALELO              │
│                 │  │█████│█████│█████│ │                                  │
│                 │  └─────┴─────┴─────┘ │                                  │
│                 └───────────┬───────────┘                                  │
│                             │                                               │
│                             ▼                                               │
│                    ┌────────────────┐                                       │
│                    │  CONCATENAR    │                                       │
│                    │   RESULTADOS   │                                       │
│                    └────────────────┘                                       │
│                                                                             │
│  TIEMPO:  TIEMPO_MAX(CHUNK_1...CHUNK_N) = PARALELO                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


GPU Utilization (tiempo):
███████████████████████████████
███████████████████████████████
███████████████████████████████
         ▲
         │
    GPU 100%
    utilizado
```

---

## Comparacion de Tiempos

```
ACTUAL (SECUENCIAL):                          OPTIMIZADO (BATCHED):
                                              
  Chunk 1: [████████████] 30s                  Batch:   [████████████] ~10s
             └─ espera ─┘
  Chunk 2:               [████████████] 30s        (todos los chunks en paralelo,
                                    └─ espera ┘    batch_size=16, GPU ~90%)
  Chunk 3:                             [████████████] 30s

  ...

  Chunk N:
                              [████████████] 30s

  ─────────────────────────────────────────────    ─────────────────────────
  TOTAL: ~128s                                    TOTAL: ~43s
```

---

## Comparacion de GPU

```
                    ACTUAL                      OPTIMIZADO
                  
GPU Memory:        ████████░░░░░░░  80%       ████████████████  95%
GPU Compute:       ▓▓▓▓░░░░░░░░░░░  30%       ████████████████  90%
                   (mucho tiempo        (paralelismo
                    ociosa)               maximizado)
```

---

## Resumen

| Métrica             | Actual   | Con Batching | Mejora    |
|---------------------|----------|--------------|-----------|
| Tiempo transcripción| 13.23s   | 2.58s        | **5.12x** |
| GPU utilization     | ~30%     | ~90%         | **3x**    |
| Chunks por pasada   | 1        | 16           | **16x**   |
| VRAM adicional      | —        | +0.5–1 GB    | —         |

> **Speedup medido: 5.12x** sobre faster-whisper secuencial en RTX 2070 Super.
> Audio de referencia: `obama_zach_16k.wav` (~6.6 min), modelo `base`, CUDA float16.
> Benchmark reproducible: `python benchmark_slice_c.py`

---

## Por que funciona?

1. **Sin batching:** cada chunk = 1 request a GPU + espera + resultado
2. **Con batching:** multiples chunks -> una sola corrida GPU -> resultados concatenados
3. **Sobrecarga minima:** overhead de agrupar chunks es mucho menor que el tiempo ahorrado
