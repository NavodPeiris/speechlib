# Diagnóstico: Speakers con volumen bajo son suprimidos por el pipeline de limpieza

## Síntoma

El pipeline de normalización + enhancement (`loudnorm` → `enhance_audio`) ignora
(suprime o silencia) speakers que hablan con volumen bajo respecto al speaker dominante.

## Causa raíz

### 1. `loudnorm` normaliza la loudness integrada del archivo completo

`torchaudio.functional.loudness()` mide **LUFS integrado** (promedio ponderado de toda
la duración). Si un speaker domina el 80% del audio, la ganancia se calibra para él.
Los speakers quietos quedan proporcionalmente igual de quietos que en el original —
no se ecualiza a nivel de speaker individual.

### 2. MossFormer2_SE_48K es un modelo de *speech enhancement*, no de separación

El modelo `MossFormer2_SE_48K` está diseñado para **eliminar ruido de fondo** y
realzar el habla del speaker principal. No distingue entre "segundo speaker" y "ruido".

Comportamiento observado:
- Speaker dominante (alto volumen) → preservado o amplificado
- Speaker quieto → SNR bajo respecto al speaker dominante → tratado como ruido → suprimido

### 3. Interacción loudnorm + enhancement

Flujo actual:
```
loudnorm(-14 LUFS global) → MossFormer2_SE_48K → enhanced.wav
```

El enhancement recibe un audio donde los speakers quietos tienen SNR bajo. El modelo
aplica su noise gate adaptativo, que suprime señales de baja energía en cada frame.
El resultado: los speakers quietos desaparecen del enhanced.wav.

## Condición de aparición

El problema ocurre cuando:
- Hay ≥ 2 speakers con diferencia de volumen significativa (≥ 6-10 dB estimados)
- El speaker quieto tiene frames con energía cercana al nivel de ruido de fondo
- O el speaker quieto habla mientras el speaker dominante también habla (solapamiento)

## Evidencia en el código

`speechlib/enhance_audio.py`:
```python
_MODEL_NAME = "MossFormer2_SE_48K"
result_audio = _clearvoice_model(
    input_path=str(state.working_path),
    output_path=str(tmp_dir),
    online_write=False,
)
```

No hay ningún control sobre el umbral de supresión del modelo. El modelo opera
como black box con sus parámetros internos de noise gate.

`speechlib/loudnorm.py`:
```python
current_lufs = torchaudio.functional.loudness(waveform, sr).item()
gain_db = TARGET_LUFS - current_lufs  # gain global, no por speaker
```

## Impacto

- Transcripción incompleta: segmentos del speaker quieto no tienen texto o tienen
  texto corrupto (transcribiendo silencio o artefactos del modelo)
- Diarización afectada: pyannote puede no detectar segmentos donde el speaker
  fue suprimido, o asignarlos al speaker dominante por vecindad temporal
- Embeddings de speaker: si el speaker quieto fue suprimido, sus embeddings
  calculados sobre `enhanced.wav` son incorrectos

## Lo que NO es el problema

- No es un bug de clipping/saturación (problema separado)
- No es un problema de formato de audio
- No es un problema de umbral de `SPEAKER_SIMILARITY_THRESHOLD`
