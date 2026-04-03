# Plan de Optimizacion del Pipeline — Uso de GPU/CPU/I/O

**Fecha:** 2026-03-22
**Hardware:** NVIDIA RTX 2070 Super (8 GB VRAM, 2560 CUDA cores)
**Audio de referencia:** 294.5s, multi-speaker, espanol
**Modelo:** large-v3-turbo (default)

---

## Pipeline actual — Diagrama

```
                         SINGLE AUDIO PIPELINE (sequential)
                         ==================================

  +-----------+  +----------+  +----------+  +----------+  +-----------+
  | convert   |->| resample |->| loudnorm |->| enhance  |->|torchaudio |
  | to wav    |  |  to 16k  |  | EBU R128 |  |MossFormer|  |  .load()  |
  |  mono     |  |          |  |          |  |  48kHz   |  |           |
  +-----------+  +----------+  +----------+  +----------+  +-----------+
     CPU          CPU           CPU         ## GPU ##       CPU/I/O
     ~0s          0.05s         0.25s        29.3s           ~0s
                                              |
                 +----------------------------+
                 v
  +------------------+   +--------------+   +------------+   +---------+
  |   diarization    | ->| absorb_micro | ->|transcribe  | ->| group + |
  | pyannote 3.1     |   | + merge_short|   |full_aligned|   |write VTT|
  |                  |   |              |   | Whisper    |   |         |
  +------------------+   +--------------+   +------------+   +---------+
     ## GPU ##              CPU                ## GPU ##       CPU/I/O
      9.2s                  ~0s                 9.1s           ~0s
```

### Timeline actual (1 audio, 48.2s total)

```
  CPU ###                                           ##               ##
  GPU           ======================== ========        =========
  I/O #                                    #                         #
  ====================================================================
  0s        10s       20s       30s   38s  40s       48s
            <--- enhance --------> <diar>     <- transcribe ->
                 29.3s              9.2s          9.1s
```

---

## Benchmark medido (SPEECHLIB_PROFILE=1, SPEECHLIB_PROFILE_KERNELS=0)

| Step | Wall time | % Total | Recurso | VRAM |
|------|-----------|---------|---------|------|
| enhance_audio | 29.3s | 60.8% | GPU+CPU | +221 MB |
| transcription | 9.1s | 18.9% | GPU | 253 MB |
| diarization | 9.2s | 19.1% | GPU | 253 MB |
| loudnorm | 0.25s | 0.5% | CPU | - |
| resample_to_16k | 0.05s | 0.1% | CPU | - |
| write_log_file | 0.003s | 0.0% | I/O | - |
| **TOTAL** | **48.2s** | | | **253 MB peak** |

RTF (Real-Time Factor): 0.16x (6x mas rapido que tiempo real)

### VRAM budget

```
  VRAM: 8 GB total
  +------------------------------------------------------+
  | enhance (MossFormer2)     221 MB                     |
  | diarization (pyannote)     32 MB                     |
  | transcription (Whisper)    ~0 MB (reutiliza espacio) |
  | TOTAL peak:              253 MB  (3% de 8 GB)        |
  | LIBRE:                 ~7.7 GB                       |
  +------------------------------------------------------+
```

### IMPORTANTE: torch.profiler causa overhead 10x

El kernel_profiler (SPEECHLIB_PROFILE_KERNELS=1) usa `torch.profiler` que causa
overhead masivo:

| Step | Sin profiler | Con profiler | Overhead |
|------|-------------|-------------|----------|
| enhance_audio | 29.3s | 465.3s | 15.9x |
| diarization | 9.2s | 37.2s | 4.0x |
| transcription | 9.1s | 178.7s | 5.6x |
| **TOTAL** | **48.2s** | **681.5s** | **9.6x** |

Siempre medir con `SPEECHLIB_PROFILE_KERNELS=0`.

---

## Dependencias reales entre pasos

```
  wav -> mono -> resample -> loudnorm ---> diarization ---> transcription
                                |                               ^
                                +---> enhance ------------------+
                                      (solo para transcription,
                                       no para diarization)
```

**Restriccion critica:** diarization necesita loudnorm porque el VAD (Voice
Activity Detection) de pyannote usa umbrales de energia. Sin normalizacion,
speakers con volumen bajo son clasificados como silencio y se pierden
irrecuperablemente.

**Enhance no es necesario para diarization** — pyannote detecta "quien habla
cuando", no necesita audio limpio. El enhance solo beneficia a Whisper (mejor
SNR -> menos errores de texto).

---

## Comparacion de modelos Whisper

| Modelo | Transcription time | Pipeline total | Calidad |
|--------|-------------------|----------------|---------|
| base | 1.8s | 40s | Mala — errores graves en espanol |
| large-v3 | 31.9s | 70.7s | Excelente — casi identica a referencia |
| large-v3-turbo | 9.1s | 48.2s | Muy buena — marginal inferior a v3 |

**Decision:** large-v3-turbo como default. large-v3 disponible via argumento.

---

## Oportunidad: overlap enhance + diarization

### Propuesta

Correr diarization sobre audio post-loudnorm (sin enhance) en un CUDA stream
paralelo mientras enhance procesa en otro stream. Cuando ambos terminan,
transcription usa audio enhanced + segmentos de diarization.

```
  PROPUESTO (overlap):
  =====================================================
  GPU stream 1: ======== enhance ================ (29s)
  GPU stream 2:     == diarization ==               (9s, sobre audio loudnorm'd)
  GPU stream 1:                    == transcription == (9s, sobre audio enhanced)
  =====================================================
  0s       9s                  29s              38s

  Ahorro: ~10s (48s -> ~38s), 20% mejora
```

### VRAM simultaneo

enhance (221 MB) + diarization (32 MB) = 253 MB — cabe en 8 GB sin problema.

### Riesgos

- CUDA kernels compitiendo por SMs podria degradar rendimiento de ambos
- Requiere dos cargas de audio (una para enhance a 48kHz, otra para diarization a 16kHz)
- Necesita `torch.cuda.Stream` y sincronizacion explicita

### Implementacion

1. Despues de loudnorm, guardar referencia al audio 16kHz
2. Lanzar enhance en GPU stream default
3. Lanzar diarization en GPU stream secundario sobre audio 16kHz
4. `stream.synchronize()` antes de transcription
5. Transcription usa audio enhanced + segmentos de diarization

---

## Multi-audio: pipeline de 3 stages

```
  CPU:    #prep1# #prep2# #prep3# #prep4#              (ThreadPool)
  GPU-1:          ====enh1==== ====enh2==== ====enh3====
  GPU-2:                  ==diar1== ==diar2== ==diar3==  (CUDA stream)
  GPU-3:                       ==trn1== ==trn2== ==trn3==
  CPU:                              #grp1# #grp2# #grp3#
```

Throughput: 1 audio cada ~29s (bottleneck = enhance) vs 48s actual.
Speedup: 1.66x para batch de audios.

Requiere `concurrent.futures.ThreadPoolExecutor` para CPU prep +
`torch.cuda.Stream` para GPU stages.

---

## Fuentes

- VAD y speakers de volumen bajo: https://picovoice.ai/blog/complete-guide-voice-activity-detection-vad/
- Speech Processing / VAD: https://speechprocessingbook.aalto.fi/Recognition/Voice_activity_detection.html
- CUDA Streams PyTorch: https://docs.pytorch.org/docs/stable/notes/cuda.html
- Pipeline parallelism con CUDA Streams: https://chaimrand.medium.com/pipelining-ai-ml-training-workloads-with-cuda-streams-bf5746449409
- Pyannote preprocessing: https://github.com/pyannote/pyannote-audio/issues/1053
- Whisper+Pyannote pipeline: https://medium.com/@rafaelgalle1/building-a-custom-scalable-audio-transcription-pipeline-whisper-pyannote-ffmpeg-d0f03f884330
