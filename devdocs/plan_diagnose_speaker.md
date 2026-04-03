# Plan: Diagnóstico mínimo — Speaker A reconocido como [unknown] en 1:42:27

## Síntoma

`Voz 260320_164522_16k_121520_es.vtt` etiqueta segmentos de Speaker A como `[unknown]`
a partir de ~01:42:17. El mismo speaker es identificado correctamente en otras partes
del archivo.

## Hipótesis (sin confirmar)

| # | Hipótesis | Observable |
|---|-----------|------------|
| H1 | Cosine similarity de Speaker A en ese tramo < threshold (0.40) | `score_agustin < 0.40` |
| H2 | Diarización asignó el tramo a un SPEAKER_TAG diferente cuyo pool de segmentos tiene baja similitud con Speaker A | speaker_tag ≠ tag principal de Speaker A |
| H3 | Segmento demasiado corto / silencio / ruido → embedding degenerado | error o score ~0 para todos |
| H4 | VTT generado con threshold distinto al actual (0.75 original) | irrelevante si se re-genera |

## Herramienta de diagnóstico

### `speechlib/tools/diagnose_speaker.py`

**Input:** audio path, timestamp HH:MM:SS, voices_folder
**Output:** tabla de cosine scores para todos los speakers vs threshold

```
python speechlib/tools/diagnose_speaker.py \
  "C:\workspace\@recordings\20260320 Patricio Renner\Voz 260320_164522.m4a" \
  01:42:17 \
  "C:\workspace\#dev\speechlib\transcript_samples\voices"
```

**Qué hace internamente:**

1. `ffmpeg -ss {timestamp} -t 30` → `temp_diag.wav` (30s alrededor del timestamp)
2. `get_embedding(temp_diag.wav)` → `test_emb`
3. Para cada speaker en `voices/` (excluyendo `_`):
   - cargar embeddings individuales por segmento
   - calcular `cosine_similarity(test_emb, emb)` por segmento
   - calcular promedio
4. Imprimir tabla: `speaker | min_score | avg_score | max_score | PASS/FAIL vs 0.40`

**Qué revela:** si H1 es correcta, `Speaker A avg < 0.40`. Si H3, todos los scores
son cercanos a 0 o NaN. Si H2, requiere inspección de diarización (fuera del scope).

## Qué NO hace este diagnóstico

- No modifica el threshold
- No re-procesa el archivo
- No analiza la diarización (qué SPEAKER_TAG fue asignado al tramo)
- No sugiere fix

## Diagnóstico extendido (si H1 confirmada)

Si el problema es el threshold, el siguiente paso sería
`speechlib/tools/calibrate_threshold.py` — procesar N archivos con speakers
conocidos y graficar la distribución de scores para encontrar el threshold
óptimo. Ese tool no está en este plan.
