# Analisis Visual: Pipeline Actual vs Batched Whisper

## Pipeline Actual ( secuencial )

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

## Pipeline Optimizado ( Batched )

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSCRIPCION BATCHED (~40 seg)                        │
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
                                              
  Chunk 1: [████████████] 30s                  Batch 1: [████████████] 10s
             └─ espera ─┘                        
  Chunk 2:               [████████████] 30s        (chunks 1-3 en paralelo)
                                    └─ espera ┘  
  Chunk 3:                             [████████████] 30s
                                                    
  ...                                              
                                                    
  Chunk N:                                        Chunk N:
                              [████████████] 30s          (chunks N-2...N en paralelo)
                                                                          
  ─────────────────────────────────────────────    ─────────────────────────
  TOTAL: ~128s                                    TOTAL: ~40s
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

| Metrica              | Actual   | Optimizado | Mejora    |
|---------------------|----------|------------|----------|
| Tiempo total        | 128 seg  | ~40 seg    | **3.2x** |
| GPU utilization     | ~30%     | ~90%       | **3x**   |
| Chunks procesados   | 1 por vez| 16 por vez | **16x**  |
| VRAM usada          | ~1 GB    | ~1.5 GB    | +0.5 GB  |

---

## Por que funciona?

1. **Sin batching:** cada chunk = 1 request a GPU + espera + resultado
2. **Con batching:** multiples chunks -> una sola corrida GPU -> resultados concatenados
3. **Sobrecarga minima:** overhead de agrupar chunks es mucho menor que el tiempo ahorrado
