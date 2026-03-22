# Diagnóstico de Rendimiento — Pipeline Speechlib

**Archivo analizado:** `transcript_samples/20260211_123242.mp3`
**Duración del audio:** 294 segundos (~4m 54s)
**Fecha:** 2026-03-22
**Hardware:** RTX 2070 Super (GPU), CPU Intel/AMD (CPU)
**Configuración:** faster-whisper large-v3, español, voices_folder activado, quantization=False

---

## Reporte de tiempos

| Paso | Tiempo | % del total | Ratio audio |
|---|---|---|---|
| `resample_to_16k` | 0.27s | 0.2% | — |
| `loudnorm` | 0.30s | 0.3% | — |
| `enhance_audio` | 29.5s | 24.8% | 0.10x |
| `diarization` | 9.4s | 7.9% | 0.032x |
| `transcription` | 79.6s | 66.9% | 0.27x |
| `write_log_file` | 0.001s | 0.0% | — |
| **TOTAL** | **118.9s** | | **0.40x** (2.5x velocidad real) |

Segmentos transcritos: 92

---

## VRAM

| Paso | Antes | Después | Delta |
|---|---|---|---|
| `enhance_audio` | 0 MB | 221 MB | +221 MB |
| `diarization` | 253 MB | 253 MB | 0 MB |
| `transcription` | 270 MB | 270 MB | 0 MB |

**Notas:**
- `enhance_audio` asigna 221 MB que no libera — MossFormer2_SE_48K queda residente en GPU.
- Diarization y transcription muestran delta 0 porque sus modelos ya estaban cargados desde runs anteriores.
- Para medir pico real de VRAM (whisper large-v3 ~3 GB) ejecutar en proceso frío.

---

## Análisis de cuellos de botella

### 1. Transcription — 67% del tiempo total (CRÍTICO)

294s de audio → 79.6s = ratio 0.27x (3.7x velocidad real solo en este paso).
92 segmentos transcritos secuencialmente. El overhead por segmento (carga de audio, llamada al modelo) se acumula. BatchedInferencePipeline(batch_size=16) ya está activo pero el cuello persiste.

**Hipótesis:**
- Los 92 segmentos son cortos en promedio (~3.2s), y el overhead de inicialización por slice puede dominar sobre la inferencia real.
- Alternativa: el modelo large-v3 es simplemente lento para este hardware.

**Acciones recomendadas:**
1. Activar `kernel_profiler` sobre el bloque transcription para separar CPU (slicing/overhead) de CUDA (inferencia).
2. Probar con `large-v2` o `medium` si la calidad lo permite — menor VRAM, menor latencia por segmento.
3. Analizar distribución de duración de segmentos — si hay muchos < 1s el merger debería fusionarlos antes de transcribir.

### 2. Enhance Audio — 25% del tiempo total

0.10x ratio (10s por cada 100s de audio). Lineal con la duración del archivo.
ClearVoice MossFormer2_SE_48K es inherentemente secuencial y costoso.

**Acciones recomendadas:**
1. Verificar con `kernel_profiler` si el cuello es I/O (leer/escribir WAV) o GPU (inferencia SE).
2. Evaluar si enhance_audio es siempre necesario — para grabaciones de alta calidad podría saltearse.

### 3. Diarization — 8% del tiempo (ACEPTABLE)

9.4s para 294s = ratio 0.032x. Pyannote 4.x está bien optimizado. No es un target de optimización prioritario.

### 4. Preprocessing — < 0.6s combinado (DESPRECIABLE)

`resample_to_16k` y `loudnorm` son irrelevantes para el rendimiento total.

---

## Distribución visual

```
transcription  ████████████████████████████████████  67%
enhance_audio  ████████████  25%
diarization    ████  8%
preprocessing  <1%
```

---

## Próximos pasos recomendados

| Prioridad | Acción | Herramienta |
|---|---|---|
| Alta | Perfilar internos de transcription (CPU vs CUDA por kernel) | `kernel_profiler.py` + `SPEECHLIB_PROFILE_KERNELS=1` |
| Alta | Probar large-v2 vs large-v3 en calidad/velocidad | benchmark manual |
| Media | Verificar si enhance_audio es I/O-bound o GPU-bound | `kernel_profiler.py` |
| Media | Analizar duración media de segmentos — ¿justifica merge más agresivo? | inspección de `common` |
| Baja | Medir pico real de VRAM en proceso frío | reiniciar proceso + step_timer |
