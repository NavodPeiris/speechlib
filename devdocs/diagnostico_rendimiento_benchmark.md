# Diagnóstico de Rendimiento — Pipeline Speechlib

**Archivo analizado:** `transcript_samples/20260211_123242.mp3`
**Duración del audio:** 294 segundos (~4m 54s)
**Hardware:** RTX 2070 Super (8GB VRAM), Windows 11
**Configuración:** faster-whisper large-v3-turbo, español, voices_folder activado, quantization=False

---

## Reporte de tiempos — BASELINE (2026-03-22)

| Paso | Tiempo | % del total |
|---|---|---|
| `resample_to_16k` | 0.27s | 0.2% |
| `loudnorm` | 0.30s | 0.3% |
| `enhance_audio` | 29.5s | 24.8% |
| `diarization` | 9.4s | 7.9% |
| `transcription` | 79.6s | 66.9% |
| **TOTAL** | **118.9s** | (2.5x velocidad real) |

Segmentos transcritos: 92

---

## Reporte de tiempos — POST-OPTIMIZACIONES con SE (2026-03-22)

Modelo: large-v3-turbo, beam_size=1, batch_size=4, MossFormer2_SE_48K

| Paso | Tiempo | % del total |
|---|---|---|
| `resample_to_16k` | 0.26s | 0.6% |
| `loudnorm` | 0.29s | 0.6% |
| `enhance_audio` | 28.3s | 62% |
| `diarization` | 9.1s | 20% |
| `transcription` | 7.7s | 17% |
| **TOTAL** | **45.7s** | (6.4x velocidad real) |

Segmentos transcritos: 90. Ahorro vs baseline: **-73.2s (-62%)**

---

## Reporte de tiempos — POST-OPTIMIZACIONES sin SE (2026-03-22)

Modelo: large-v3-turbo, beam_size=1, batch_size=4, skip_enhance=True

| Paso | Tiempo | % del total |
|---|---|---|
| `resample_to_16k` | 0.26s | 1.5% |
| `loudnorm` | 0.29s | 1.7% |
| `diarization` | 9.1s | 52% |
| `transcription` | 7.7s | 44% |
| **TOTAL** | **17.4s** | **(17x velocidad real)** |

Segmentos transcritos: 88. Ahorro vs baseline: **-101.5s (-85%)**

---

## Historial de optimizaciones

### Slice 1 — Full-file transcription con timestamp alignment
- **Impacto:** transcription 79.6s → ~24s (-55s)
- **Antes:** 92 llamadas a `batched.transcribe()`, batch=1 efectivo por segmento corto.
- **Despues:** 1 sola llamada con audio completo, batch_size=4 real.
- **Mapeo:** whisper segments alineados a diarization por overlap temporal.
- **Commit:** `56cfff2`

### Slice 2 — Cache pyannote pipeline
- **Impacto:** -2-4s en llamadas subsecuentes (startup)
- **Despues:** `@lru_cache(maxsize=1)` en `_get_diarization_pipeline()`.
- **Commit:** `6ba6bf8`

### Slice 3 — enhance_audio online_write=False
- **Impacto:** ~0s (cuello era CPU/Python, no I/O)
- **Nota:** hipótesis sobre I/O sincrono era incorrecta para este hardware.
- **Commit:** `2e4b3f9`

### Slice A — large-v3-turbo en lugar de large-v3
- **Impacto:** transcription ~2.2s (-9%)
- **Nota:** beneficio menor de lo esperado — con full-file batching, encoder domina; decoder turbo no ayuda tanto.
- **Commit:** (ver git log)

### Slice B — beam_size=1 (greedy)
- **Impacto:** transcription 24.3s → 7.7s (-69%)
- **Antes:** beam_size=5 multiplicaba trabajo del decoder 5x.
- **Commit:** `6089089`

### Slice C — FRCRN_SE_16K (DESCARTADO)
- **Hipótesis:** modelo 16kHz nativo sería más rápido que 48kHz.
- **Benchmark:** FRCRN_SE_16K = 57.4s vs MossFormer2_SE_48K = 28.3s — 2x más lento.
- **Causa probable:** arquitectura FRCRN tiene mayor costo computacional.

### Slice E — skip_enhance=True
- **Impacto:** -28.3s (-62% del total con SE)
- **Tradeoff:** calidad de transcripción sin SE pendiente de validación A/B.
- **Commit:** `585f304`

---

## Análisis del nuevo bottleneck (sin SE)

```
diarization  ████████████████████████████  52%
transcription  ███████████████████████████  44%
preprocessing  █  4%
```

### Diarization — 52% (9.1s, ratio 0.031x)

Pyannote speaker-diarization-3.1 en GPU. Tiempo estable en todos los runs.
**No hay optmizaciones obvias disponibles** sin cambiar el modelo de diarización.
Posibles alternativas:
- `pyannote/speaker-diarization-community-1` (no probado)
- Reducir número de speakers esperados si se conoce de antemano

### Transcription — 44% (7.7s, ratio 0.026x)

large-v3-turbo + beam_size=1 + batch_size=4. Muy optimizado ya.
Posibles mejoras marginales:
- batch_size=8 si VRAM lo permite (liberando MossFormer2 primero)
- `medium` o `large-v2` si calidad lo permite (~2-4s estimado)

---

## Estado del pipeline (skip_enhance=False, modo producción)

```
enhance_audio  ████████████████████████████████████  62%
diarization    ████████████████  20%
transcription  ████████████  17%
preprocessing  █  1%
```

Cuello crítico: **enhance_audio (28.3s)**. Para reducirlo:
- Ningún modelo ClearVoice alternativo probado fue más rápido
- `skip_enhance=True` elimina el cuello por completo (decisión de calidad vs latencia)

---

## Próximos pasos recomendados

| Prioridad | Acción | Comando |
|---|---|---|
| Alta | Validar calidad A/B con/sin SE en audio de reunión | Revisar output/*.txt |
| Media | Probar MossFormerGAN_SE_16K si calidad sin SE es insuficiente | benchmark manual |
| Baja | Medir pico real VRAM proceso frío | reiniciar proceso + profile_run.py |
| Baja | Evaluar batch_size=8 tras liberar VRAM de ClearVoice | profile_run.py post-release |
