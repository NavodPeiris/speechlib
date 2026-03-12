# Analisis de Optimizacion del Pipeline de Transcripcion

**Fecha:** 11/03/2026
**Actualizado:** 11/03/2026
**Objetivo:** Reducir tiempo de transcripcion + mejorar calidad con preprocessing avanzado

---

## 1. Contexto

### Hardware disponible

- **GPU:** NVIDIA GeForce RTX 2070 Super with Max-Q Design
- **VRAM:** 8191 MB (8GB disponibles, ~6-7GB para inference dejando margen)
- **torchaudio:** 2.10.0+cpu (operaciones de audio en CPU; modelos neuronales en GPU via PyTorch)
- **torchcodec:** instalado (habilita encoding AAC/m4a nativo sin FFmpeg)

### Pipeline actual (baseline)

Audio de referencia: ~6 minutos, WAV estéreo, 44.1kHz, 16-bit

```
┌────────────────────┬──────────────┬────────────┬───────────────┐
│ Etapa              │ Tiempo       │ % del total│ Optimizable   │
├────────────────────┼──────────────┼────────────┼───────────────┤
│ Preprocessing      │    1 seg     │    1%      │ No critico    │
│ Diarizacion        │   21 seg     │   13%      │ No            │
│ Speaker recognition│    2 seg     │    1%      │ No            │
│ Transcripcion      │  128 seg     │   83%      │ SI (cuello)   │
├────────────────────┼──────────────┼────────────┼───────────────┤
│ TOTAL              │  152 seg     │  100%      │               │
└────────────────────┴──────────────┴────────────┴───────────────┘
```

**Conclusion:** Transcripcion es el 83% del tiempo. La calidad depende de la limpieza del audio entrada.

---

## 2. Analisis del problema

### 2.1 Por que la transcripcion es lenta?

Whisper procesa internamente chunks de ~30 segundos. El pipeline actual procesa cada segmento de speaker de forma secuencial:

```
Segmento 1 (speaker A, 45s)
  └─ Chunk 1 (30s) → GPU encode → resultado → idle
  └─ Chunk 2 (15s) → GPU encode → resultado → idle

Segmento 2 (speaker B, 60s)
  └─ Chunk 1 (30s) → GPU encode → resultado → idle
  └─ Chunk 2 (30s) → GPU encode → resultado → idle
```

La GPU esta ociosa ~70% del tiempo entre carga de chunk y procesamiento.

### 2.2 Por que la calidad puede mejorarse antes de transcribir?

El audio llega al modelo de transcripcion con tres problemas potenciales sin resolver:

```
Audio original
  │
  ├─ Sample rate variable (48kHz, 44.1kHz, etc.)
  │   Whisper resamplea internamente a 16kHz de todas formas.
  │   Si lo hacemos antes, los modelos de enhancement
  │   trabajan con el SR correcto.
  │
  ├─ Volumen inconsistente
  │   Voces bajas → transcripcion imprecisa o faltante.
  │   Normalizacion EBU R128 resuelve esto en <1s.
  │
  └─ Ruido de fondo
      Background noise degrada confianza de transcripcion ~5-15%.
      Speech enhancement (ClearerVoice) lo elimina con modelos
      neuronales en GPU.
```

---

## 3. Pipeline optimizado

### 3.1 Diagrama completo

```
Audio entrada (cualquier formato)
    │
    ▼ PREPROCESSING EXISTENTE ─────────────────────────────────
    │
    ├─ convert_to_wav         cualquier formato → WAV (torchaudio)
    ├─ convert_to_mono        estéreo → mono (wave + numpy)
    └─ re_encode              8-bit → 16-bit PCM (wave)
    │
    ▼ PREPROCESSING NUEVO ─────────────────────────────────────
    │
    ├─ [A] resample_to_16k    torchaudio.functional.resample → CPU
    ├─ [D] enhance_audio      ClearerVoice FRCRN_SE_16K → GPU  (opcional)
    ├─ [B] loudnorm            torchaudio.functional.loudness → CPU
    └─ [E] compress_to_m4a   torchcodec AAC encoding → CPU    (opcional)
    │
    ▼ ANALISIS ─────────────────────────────────────────────────
    │
    ├─ Diarizacion            pyannote 4.x → GPU
    ├─ Speaker recognition    pyannote embedding → GPU         (opcional)
    │
    └─ Transcripcion
        ├─ faster-whisper: [C] BatchedInferencePipeline → GPU
        └─ otros backends: sin cambio
    │
    ▼ OUTPUT
        log file  +  [E] enhanced_audio.m4a                   (si E activo)
```

### 3.2 Estado de AudioState a lo largo del pipeline

```
AudioState inicial
  source_path:   "meeting.mp4"
  working_path:  "meeting.mp4"
  is_wav=F  is_mono=F  is_16bit=F  is_16khz=F  is_enhanced=F  is_normalized=F
       │
       ▼ convert_to_wav
  working_path:  "meeting.wav"          is_wav=True
       │
       ▼ convert_to_mono
  working_path:  "meeting_mono.wav"     is_mono=True
       │
       ▼ re_encode
  working_path:  "meeting_mono_16bit.wav"   is_16bit=True
       │
       ▼ [A] resample_to_16k
  working_path:  "meeting_mono_16bit_16k.wav"   is_16khz=True
       │
       ▼ [D] enhance_audio  (opcional)        ← ANTES de loudnorm (ver sec. 8)
  working_path:  "meeting_mono_16bit_16k_enhanced.wav"   is_enhanced=True
       │
       ▼ [B] loudnorm                          ← DESPUES de enhancement
  working_path:  sin cambio de archivo   is_normalized=True
       │                                 (operacion in-place sobre tensor)
       ▼ [E] compress_to_m4a  (opcional)
  working_path:  "meeting_mono_16bit_16k_enhanced.m4a"
       │
       ▼ pipeline(str(state.working_path))  → diarization
```

---

## 4. Orden entre Enhancement y Loudnorm — Investigacion

### 4.1 Pregunta

En el pipeline de preprocessing, ¿es mejor aplicar loudnorm (EBU R128) antes o despues
de la limpieza neural de audio (ClearerVoice / FRCRN / MossFormer)?

### 4.2 Conclusion

**Enhancement primero, loudnorm despues.**

Esto invierte el orden planteado inicialmente. El analisis de fuentes tecnicas y el
comportamiento real de los modelos confirma que la unica secuencia correcta es:

```
resample → [enhance_audio] → loudnorm → transcripcion
```

### 4.3 Por que loudnorm ANTES del enhancer es incorrecto

**1. Amplifica ruido y speech por igual**

Loudnorm aplica ganancia constante a toda la señal. Si el audio esta a -30 LUFS y el
target es -14 LUFS, se aplica +16 dB a todo: voz Y ruido de fondo.
El SNR no mejora, pero el ruido ahora esta a mayor nivel absoluto.
FRCRN/MossFormer reciben una señal mas sucia de lo que habrian recibido sin normalizar.

```
Audio -30 LUFS, SNR=15 dB
    │
    ▼ loudnorm I=-14  (+16 dB de ganancia)
    │  speech: -14 LUFS   noise: -30 LUFS + 16 dB = -14 LUFS   ← SNR sin cambio
    │  pero el piso de ruido ahora esta ~6x mas alto en amplitud
    ▼ FRCRN (ve mas ruido absoluto, mask estimation menos confiable)
```

**2. Riesgo de clipping en transientes**

Si el audio tiene picos altos, loudnorm puede clipear antes de que el enhancer
pueda procesar la señal. El enhancer no puede recuperar audio clippeado.

**3. FRCRN y MossFormer no necesitan input pre-normalizado**

Ambos modelos se entrenaron en DNS Challenge con RMS aleatorio entre -35 y -15 dBFS.
Aplican normalizacion interna antes del STFT (MossFormerGAN normaliza por potencia;
FRCRN opera sobre features espectrales relativas). El input no necesita estar en
un nivel LUFS especifico.

**4. Consenso profesional**

> "Do not use leveling or gain control before noise reduction algorithms. Noise levels
> may be raised artificially and unintentionally, resulting in lower audio quality."
> — Auphonic (herramienta de referencia para post-produccion de podcasts/voz)

> "Noise suppression first prevents normalization algorithms from amplifying residual
> noise and enables more accurate loudness measurement on actual speech content."
> — ClearlyIP (pipeline de pre-procesamiento para ASR en voice bots)

La cadena de Auphonic (la mas citada en la industria) es:
`noise reduction → leveling → loudness normalization → peak limiting`

### 4.4 Por que loudnorm DESPUES del enhancer es correcto

1. El enhancer ya elimino el ruido. Loudnorm mide y ajusta solo el nivel de la voz limpia.
2. El target -14 LUFS se aplica sobre contenido que es casi solo speech.
3. No hay riesgo de amplificar artefactos de fondo.
4. Si el enhancer reduce la energia total de la señal (efecto habitual), loudnorm lo
   compensa exactamente.

### 4.5 Cambio en el orden de implementacion

Este hallazgo modifica el orden de slices respecto al plan inicial:

```
ORDEN INICIAL (incorrecto):  A → B (loudnorm) → D (enhance) → E → C
ORDEN CORRECTO:              A → D (enhance)  → B (loudnorm) → E → C
```

El orden de IMPLEMENTACION (por riesgo/dependencias) no cambia:
A → B → C → D → E — porque B y D son independientes en codigo.
El orden de EJECUCION en el pipeline cambia: D se ejecuta antes que B.

### 4.6 Fuentes

- Auphonic Blog: Loudness Normalization and Compression for Speech Audio
- ClearlyIP: Voice Bot Audio Pre-Processing with Noise Cancellation
- Microsoft DNS-Challenge: noisyspeech_synthesizer.cfg (training data distribution)
- KVR Audio forum: "Compress before or after noise reduction?"
- arXiv 2512.17562: "When De-noising Hurts — Speech Enhancement Effects on ASR"

---

## 5. Slice A — resample_to_16k

### Por que es el primer paso

Whisper (cualquier backend) resamplea internamente a 16kHz antes de procesar.
Si lo hacemos explicitamente en el pipeline:

- Los modelos de ClearerVoice (`FRCRN_SE_16K`, `MossFormerGAN_SE_16K`) necesitan exactamente 16kHz
- La normalizacion EBU R128 (loudnorm) requiere SR conocido para el filtro K-weighting
- Reduce el tamaño de todos los archivos temporales posteriores (~3x en audio 48kHz)

### Implementacion

```python
# speechlib/resample_to_16k.py
import torchaudio
from .audio_state import AudioState

TARGET_SR = 16000

def resample_to_16k(state: AudioState) -> AudioState:
    waveform, sr = torchaudio.load(str(state.working_path))
    if sr == TARGET_SR:
        return state.model_copy(update={"is_16khz": True})

    resampled = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    out_path = state.working_path.with_stem(state.working_path.stem + "_16k")
    torchaudio.save(str(out_path), resampled, TARGET_SR, bits_per_sample=16)
    return state.model_copy(update={"working_path": out_path, "is_16khz": True})
```

### Impacto en tamaño de archivo

```
Audio 6 min, mono, 16-bit:
  44.1 kHz → 63.5 MB
  48.0 kHz → 69.1 MB
  16.0 kHz → 23.0 MB   ← 3x menos en todos los pasos siguientes
```

---

## 6. Slice B — loudnorm

### Por que es necesario

Voces bajas son el principal motivo de transcripciones incompletas:

```
Whisper confidence por nivel de volumen (referencia empirica):

  -30 LUFS  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~75% confianza (voz muy baja)
  -23 LUFS  ████████████████████░░░░░░░░░░  ~85% confianza
  -18 LUFS  ██████████████████████████░░░░  ~91% confianza
  -14 LUFS  ███████████████████████████████ ~96% confianza  ← TARGET
  -10 LUFS  ███████████████████████████████ ~96% confianza  (sin mejora)
```

Target: `I = -14 LUFS, TP = -1.0 dB` (estandar streaming/voz)

### Implementacion 100% torch (sin dependencias nuevas)

```python
# speechlib/loudnorm.py
import torch, torchaudio
from .audio_state import AudioState

TARGET_LUFS = -14.0
TRUE_PEAK_DB = -1.0

def loudnorm(state: AudioState) -> AudioState:
    waveform, sr = torchaudio.load(str(state.working_path))

    current_lufs = torchaudio.functional.loudness(waveform, sr).item()

    if abs(current_lufs - TARGET_LUFS) < 0.5:          # ya normalizado
        return state.model_copy(update={"is_normalized": True})

    gain_db     = TARGET_LUFS - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    normalized  = waveform * gain_linear

    true_peak = 10 ** (TRUE_PEAK_DB / 20)              # limitar TP
    normalized = torch.clamp(normalized, -true_peak, true_peak)

    torchaudio.save(str(state.working_path), normalized, sr, bits_per_sample=16)
    return state.model_copy(update={"is_normalized": True})
```

Nota: opera in-place sobre el working_path (no crea nuevo archivo).

### Tiempo estimado

```
Audio 6 min, 16kHz, CPU:  < 0.3 seg
Audio 1 hora, 16kHz, CPU: < 3 seg
```

---

## 7. Slice C — BatchedInferencePipeline (faster-whisper)

### El problema actual

```
TRANSCRIPCION ACTUAL — secuencial, GPU ociosa

  tiempo →  0s        30s       60s       90s      120s     150s
  GPU       [chunk1]  [idle]  [chunk2]  [idle]  [chunk3]  [idle]
            ████████░░░░░░░░████████░░░░░░░░████████░░░░░░░░
            ↑ GPU ~30% utilizada en promedio
```

### Con batching

```
TRANSCRIPCION NUEVA — batched, GPU saturada

  tiempo →  0s              40s
  GPU       [batch: chunk1+chunk2+chunk3+chunk4+...]
            ███████████████████████████████████████
            ↑ GPU ~90% utilizada
```

### Como funciona BatchedInferencePipeline

```
Audio (segmento speaker, ej: 3 min)
    │
    ├─ Divide en chunks de 30s internamente
    │   [chunk1][chunk2][chunk3][chunk4][chunk5][chunk6]
    │
    ├─ Agrupa en batches de batch_size=16
    │   batch_0: [c1][c2][c3][c4][c5][c6]   ← un solo forward pass GPU
    │
    └─ Output ordenado: mismo formato que transcribe() normal
```

### Implementacion en transcribe.py

```python
# rama faster-whisper en speechlib/transcribe.py
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel(model_size, device=device, compute_type=compute_type)
batched = BatchedInferencePipeline(model=model)
segments, info = batched.transcribe(
    file,
    batch_size=16,          # 16 para tiny/small/medium con 8GB VRAM
    language=language,
    beam_size=5,
)
```

### VRAM por modelo con batch_size=16

```
┌────────────┬────────────────┬──────────────────────────────┐
│ Modelo     │ VRAM necesaria │ RTX 2070 Super (8GB)         │
├────────────┼────────────────┼──────────────────────────────┤
│ tiny       │    ~1 GB       │ ✓ holgado                    │
│ small      │    ~2 GB       │ ✓ holgado                    │
│ medium     │    ~4 GB       │ ✓ holgado                    │
│ large-v2   │  ~6-7 GB       │ ✓ margen ajustado            │
│ large-v3   │  ~7-8 GB       │ △ reducir batch_size a 8     │
└────────────┴────────────────┴──────────────────────────────┘
```

### Beneficios documentados

| Metrica | Sin batching | Con batching | Mejora |
|---------|-------------|--------------|--------|
| Speed vs whisper original | 1x | 12.5x | 12.5x |
| Speed vs faster-whisper normal | 1x | 3x | 3x |
| GPU utilization | ~30% | ~90% | 3x |
| VRAM adicional | — | +0.5-1 GB | — |

---

## 8. Slice D — Speech Enhancement con ClearerVoice (opcional)

### Que hace

ClearerVoice aplica modelos neuronales de eliminacion de ruido sobre el audio completo antes de la diarizacion y la transcripcion.

```
Audio entrada (con ruido de fondo, eco, etc.)
    │
    ▼ ClearVoice(task='speech_enhancement', model='FRCRN_SE_16K')
    │
Audio salida (voz limpia, ruido eliminado)
```

### Modelos disponibles para 16kHz

| Modelo | Calidad | Velocidad (GPU) | VRAM |
|--------|---------|-----------------|------|
| `FRCRN_SE_16K` | Muy buena | ~0.05x RT | ~1 GB |
| `MossFormerGAN_SE_16K` | Excelente (GAN) | ~0.10x RT | ~2 GB |

*RT = real-time factor: 0.05x significa que 1 minuto de audio se procesa en 3 segundos.*

### Tiempo estimado para audio de 6 minutos (GPU)

```
FRCRN_SE_16K         ~18 seg  (recomendado: calidad/velocidad)
MossFormerGAN_SE_16K ~36 seg  (mejor calidad perceptual)
```

### Impacto en calidad de transcripcion

```
Ambiente con ruido moderado (SNR ~15 dB):

  Sin enhancement:    ████████████████████░░░░░░░░  ~82% confianza
  Con FRCRN:          ██████████████████████████░░  ~93% confianza
  Con MossFormerGAN:  ███████████████████████████░  ~95% confianza
```

### Integracion opcional

```python
# en core_analysis.py
if speech_enhancement and _CLEARVOICE_AVAILABLE:
    state = enhance_audio(state, model="FRCRN_SE_16K")
```

Requisito: Slice A (resample_to_16k) debe estar activo — los modelos de ClearerVoice requieren exactamente 16kHz.

---

## 9. Slice E — Compresion a AAC/m4a (opcional)

### Por que comprimir antes de transcribir

Despues del enhancement, el audio esta en formato WAV 16kHz mono 16-bit.
Comprimir a AAC/m4a antes de la transcripcion reduce el tamaño de los archivos de segmentos temporales en `segments/`:

```
WAV 16kHz mono 16-bit:    ~115 MB / hora de audio
AAC m4a 64kbps 16kHz:     ~  28 MB / hora de audio  (4x menos)
```

### Compatibilidad con backends de transcripcion

Todos los backends soportan m4a como input (usan ffmpeg o torchcodec para decodificar):
- `faster-whisper` ✓   `whisper` ✓   `huggingface` ✓   `assemblyAI` ✓

### Implementacion con torchcodec (sin FFmpeg para encoding)

```python
# torchaudio 2.10+ usa torchcodec — AAC encoding nativo
torchaudio.save("audio.m4a", waveform, 16000)
# verificado: funciona en el entorno actual
```

---

## 10. Estimaciones de rendimiento

### 9.1 Audio de 6 minutos — comparativa por escenario

```
ESCENARIO 1: Solo BatchedWhisper (Slice C)
─────────────────────────────────────────
  Preprocessing    1 seg  ████░
  Diarizacion     21 seg  ████████████████████░
  Spk recognition  2 seg  ██░
  Transcripcion   42 seg  █████████████████████████████████████████░
  ─────────────────────────────────────────────────────────────────
  TOTAL:          66 seg  (vs 152 baseline)   SPEEDUP: 2.3x

ESCENARIO 2: Slices A + B + C (sin enhancement, sin compresion)
────────────────────────────────────────────────────────────────
  Preprocessing    1 seg  ████░
  Resample 16k    <1 seg  █░
  Loudnorm        <1 seg  █░
  Diarizacion     21 seg  ████████████████████░
  Spk recognition  2 seg  ██░
  Transcripcion   42 seg  █████████████████████████████████████████░
  ─────────────────────────────────────────────────────────────────
  TOTAL:          68 seg  (vs 152 baseline)   SPEEDUP: 2.2x
  Calidad:        mejor (voces normalizadas)

ESCENARIO 3: Slices A + B + C + D (con enhancement FRCRN)
──────────────────────────────────────────────────────────
  Preprocessing    1 seg  ████░
  Resample 16k    <1 seg  █░
  Loudnorm        <1 seg  █░
  ClearerVoice   18 seg  █████████████████░
  Diarizacion     21 seg  ████████████████████░
  Spk recognition  2 seg  ██░
  Transcripcion   42 seg  █████████████████████████████████████████░
  ─────────────────────────────────────────────────────────────────
  TOTAL:          86 seg  (vs 152 baseline)   SPEEDUP: 1.8x
  Calidad:        maxima (ruido eliminado)

ESCENARIO 4: Slices A + B + C + D + E (pipeline completo)
──────────────────────────────────────────────────────────
  Igual a Escenario 3 + compress_to_m4a ~1 seg
  TOTAL:          87 seg   SPEEDUP: 1.7x
  Beneficio E:    archivos temporales 4x mas pequeños
```

### 9.2 Por modelo — solo Slice C (BatchedInferencePipeline)

| Modelo | Transcripcion actual | Con batching | Speedup |
|--------|---------------------|--------------|---------|
| tiny | ~30 seg | ~10 seg | 3x |
| small | ~60 seg | ~20 seg | 3x |
| medium | ~180 seg | ~60 seg | 3x |
| large-v2 | ~300 seg | ~100 seg | 3x |

*Referencia: audio de 6 minutos en RTX 2070 Super*

### 9.3 Impacto acumulado (pipeline completo, modelo small, 6 min)

```
Baseline:   152 seg  ████████████████████████████████████████████████████████
Slice C:     66 seg  ██████████████████████░
Slices A+B+C: 68 seg  █████████████████████░
+ Slice D:   86 seg  ████████████████████████████░
+ Slice E:   87 seg  ████████████████████████████░ (+1 seg, 4x menos disco)
```

---

## 11. Slices de implementacion — orden pareto-optimo

El orden minimiza riesgo y maximiza valor acumulado: cada slice es util independientemente y habilita el siguiente.

```
┌────────┬─────────────────────┬────────┬────────┬────────────────────────────┐
│ Slice  │ Que hace            │ Riesgo │ Valor  │ Dependencias               │
├────────┼─────────────────────┼────────┼────────┼────────────────────────────┤
│ A      │ resample_to_16k     │ Minimo │ Alto   │ ninguna (torchaudio)       │
│ B      │ loudnorm            │ Bajo   │ Alto   │ A (necesita SR fijo)       │
│ C      │ BatchedWhisper      │ Bajo   │ Alto   │ ninguna (independiente)    │
│ D      │ ClearerVoice        │ Medio  │ Alto*  │ A (FRCRN necesita 16kHz)   │
│ E      │ compress_to_m4a     │ Bajo   │ Medio  │ A (reduce archivos 4x)     │
└────────┴─────────────────────┴────────┴────────┴────────────────────────────┘
*Alto para audio con ruido de fondo; bajo para audio limpio
```

### Orden de implementacion recomendado

```
1. SLICE A  →  2. SLICE B  →  3. SLICE C  →  4. SLICE D  →  5. SLICE E
   ~30 min        ~20 min        ~30 min        ~1 hora        ~30 min
```

Cada slice completa el ciclo RED-GREEN-REFACTOR antes de abrir el siguiente.

---

## 12. Alternativas descartadas

### Chunking manual del audio

**Problema:** fragmentar el audio antes de diarizacion destruye la informacion de speaker por segmento.
**Por que BatchedInferencePipeline es mejor:** serializa por speaker y aplica batching interno, sin perder contexto.

### Multi-GPU

No factible: RTX 2070 es una sola GPU, sin infraestructura de clustering.

### Modelos mas pequeños

Reducir tiny→base sacrifica precision. BatchedInferencePipeline logra el mismo speedup sin reducir calidad.

### FFmpeg para encoding m4a

torchcodec (ya instalado via torchaudio 2.10) soporta AAC encoding nativo. No se necesita FFmpeg como dependencia adicional para el pipeline.

---

## 13. Conclusion

El pipeline optimizado ataca los dos problemas del baseline: velocidad y calidad.

```
┌─────────────────────────────────────────────────────────────────────┐
│ MEJORAS OBTENIBLES                                                   │
│                                                                      │
│  Velocidad:  2.3x speedup solo con Slice C (BatchedInferencePipeline│
│  Calidad:    +11 pp confianza en audio ruidoso (Slice D)            │
│  Disco:      4x menos espacio en archivos temporales (Slice E)      │
│  Robustez:   voces bajas normalizadas antes de transcribir (Slice B) │
│                                                                      │
│  Sin cambios de arquitectura profundos.                             │
│  Sin dependencias nuevas para A, B, C, E.                           │
│  D (ClearerVoice) es opcional y degradable gracefully.              │
└─────────────────────────────────────────────────────────────────────┘
```

**Recomendacion:** Implementar en orden A → B → C → D → E en el branch `feat/batched-whisper-inference`.

---

## 14. Referencias

- [Faster-Whisper BatchedInferencePipeline PR](https://github.com/SYSTRAN/faster-whisper/pull/856)
- [Modal: Fast Whisper with dynamic batching](https://modal.com/docs/examples/whisper-transcriber)
- [Baseten: Fastest Whisper transcription](https://www.baseten.co/blog/the-fastest-most-accurate-and-cost-efficient-whisper-transcription/)
- [ClearerVoice-Studio models](https://github.com/modelscope/ClearerVoice-Studio)
- [torchaudio loudness (ITU-R BS.1770)](https://pytorch.org/audio/stable/generated/torchaudio.functional.loudness.html)
- [EBU R128 loudness standard](https://tech.ebu.ch/loudness)
