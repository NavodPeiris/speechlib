# Plan de Implementacion: BatchedInferencePipeline para Whisper

## Objetivo
Reducir tiempo de transcripcion en ~3x usando batched inference de faster-whisper sobre GPU.

**Hardware objetivo:** NVIDIA RTX 2070 Super (8GB VRAM)
**Speedup esperado:** 3x sobre faster-whisper secuencial (12.5x sobre whisper original)

---

## Contexto tecnico

### El problema actual
- El pipeline actual usa `transcribe()` secuencial de faster-whisper
- Whisper procesa ~30 segundos por chunk internamente
- Cada chunk se procesa individualmente: GPU subutilizada

### La solucion
`BatchedInferencePipeline` de faster-whisper procesa multiples chunks en paralelo en la misma corrida de GPU:
- Mantiene la misma calidad de transcripcion
- No requiere cambios en la arquitectura del pipeline
- Compatible con modelos tiny/small/medium/large-v2

---

## Analisis de cambios

### 1. Modificar `speechlib/wav_segmenter.py`

**Cambio principal:** Reemplazar:

```python
# ACTUAL (secuencial)
segments, info = model.transcribe(audio, language=language, ...)
```

Por:

```python
# NUEVO (batched)
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe(audio, batch_size=16, language=language, ...)
```

**Parametros clave:**
- `batch_size`: 16 para 8GB VRAM (ajustar para otros modelos)
- `max_num_frames`: controlar longitud maxima por chunk

### 2. Compatibilidad

**Metodos afectados:**
- `Transcriptor.whisper()` - usa `wav_segmenter.py`
- `Transcriptor.faster_whisper()` - ya usa faster-whisper
- `Transcriptor.custom_whisper()` - verificar compatibilidad
- `Transcriptor.huggingface_model()` - no aplica (usa transformers)

**Backward compatibility:** 100% - mismo output, solo mas rapido

---

## Implementacion por pasos

### SLICE 1: Modificar `wav_segmenter.py` (~30 min)

1. Importar `BatchedInferencePipeline`
2. Crear wrapper que use batched inference cuando este disponible
3. Mantener fallback a transcribe() normal si falla
4. Agregar parametro `use_batching=True` (default True)

### SLICE 2: Testing (~15 min)

1. Comparar tiempo de ejecucion antes/despues con mismo audio
2. Verificar que output sea identico (mismos timestamps y texto)
3. Testear con diferentes modelos (tiny, small, medium)

### SLICE 3: Documentacion (~15 min)

1. Actualizar README con nueva opcion
2. Documentar parametros de batch_size

---

## Estimacion de mejora

| Modelo | Tiempo actual | Tiempo esperado | Speedup |
|--------|--------------|----------------|---------|
| tiny   | ~30 seg      | ~10 seg        | 3x      |
| small  | ~60 seg      | ~20 seg        | 3x      |
| medium | ~180 seg     | ~60 seg        | 3x      |

*Basado en benchmark de faster-whisper con batch_size=16*

---

## VRAM por modelo (batch_size=16)

| Modelo      | VRAM necesaria |
|-------------|---------------|
| tiny        | ~1 GB         |
| small       | ~2 GB         |
| medium      | ~4 GB         |
| large-v2    | ~6-7 GB       |

RTX 2070 Super (8GB): puede usar hasta large-v2 con batch_size moderado.

---

## Fallbacks y edge cases

1. **Si batched inference falla:** usar transcribe() normal
2. **Si VRAM insuficiente:** reducir batch_size o usar modelo mas pequeno
3. **Si audio muy corto:** overhead de batching no justifica, usar normal
4. **Transcription hotwords, language detection:** compatibles

---

## Definition of Done

- [ ] `wav_segmenter.py` modificado para usar BatchedInferencePipeline
- [ ] Mismo output que version anterior (verificable con test)
- [ ] Tiempo de transcripcion reducido ~3x
- [ ] README actualizado
- [ ] Branch mergeado a main
