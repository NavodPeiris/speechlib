# Plan: Carpeta de artefactos `.{audio_stem}/`

## Estado deseado

Dado un audio `C:\recordings\Voz 260320.m4a`, todos los artefactos van a:

```
C:\recordings\.Voz 260320\
  source.wav              # convert_to_wav
  16k.wav                 # resample (working file)
  enhanced.wav            # enhance_audio
  compressed.m4a          # compress_audio
  diarization.rttm        # pyannote (cache)
  embeddings.npz          # speaker embeddings (cache)
  transcript_es.vtt       # output final
  relabeled_es.vtt        # relabel_vtt output
  unknown_speakers/       # clips de desconocidos
```

Cache por existencia de archivo: `if (artifacts_dir / "enhanced.wav").exists(): skip`.

---

## Diagnóstico de brechas

### B1 — Preprocessing disperso junto al source

| Módulo | Escribe | Dónde |
|--------|---------|-------|
| `convert_to_wav.py` | `{stem}.wav` | junto al source |
| `convert_to_mono.py` | `{stem}_mono.wav` | junto al source |
| `re_encode.py` | `{stem}_16bit.wav` | junto al source |
| `resample_to_16k.py` | `{stem}_16k.wav` | junto al source |
| `loudnorm.py` | sobreescribe working_path in-place | — |

Contamina la carpeta del usuario con 4-5 archivos intermedios.

### B2 — `enhance_audio` crea su propio patrón de subcarpeta

Escribe a `{stem}_enhanced_out/MossFormer2_SE_48K/{name}` — patrón propio de ClearVoice,
inconsistente con el estándar propuesto.

### B3 — `compress_audio` escribe junto al source

`state.source_path.with_suffix(".m4a")` — archivo perdido entre los del usuario.

### B4 — Temp dirs hardcodeados a CWD

| Módulo | Dir | Riesgo |
|--------|-----|--------|
| `speaker_recognition.py` | `./temp/` | colisión en runs concurrentes |
| `wav_segmenter.py` | `./segments/` | colisión en runs concurrentes |

No es cache reutilizable; se borran inmediatamente.

### B5 — VTT va a `log_folder` con timestamp en nombre

`write_log_file.py` genera `{stem}_{HHMMSS}_{lang}.vtt` en `log_folder`.
Múltiples VTTs por audio sin saber cuál es el vigente.

### B6 — Cero lógica de cache/resume

Ningún módulo verifica si un artefacto previo existe antes de recomputar.
Re-run de 135 min repite: enhance (~40 min GPU), diarización (~2 min GPU),
transcripción (~5 min).

### B7 — Tools escriben productos en ubicaciones arbitrarias

| Tool | Producto | Dónde escribe | Problema |
|------|----------|---------------|----------|
| `relabel_vtt.py` | `_relabeled.vtt`, `_relabeled_padded.vtt` | junto al VTT original (carpeta del usuario) | Contamina carpeta con variantes |
| `diagnose_speaker.py` | `diag_*.wav` temp | `tempfile.mktemp()` en system temp | No reutilizable como cache |
| `process_recordings.py` | `*_sample.wav` en temp, unknown clips | system temp (borrado) + `voices/_unknown/` hardcodeado | Samples perdidos, unknown_output_dir fijo |
| `extract_unknown_speakers.py` | `{tag}_{stem}/segment_NN.wav` | `output_dir` param (default: `voices/../_unknown`) | Fuera del contexto del audio procesado |
| `batch_process.py` | delega a `core_analysis` + `extract_unknown_speakers` | `log_folder` = carpeta del audio, unknown = `voices/_unknown` | Outputs dispersos en 2+ ubicaciones |

**Resumen B7:** cada tool decide independientemente dónde poner sus productos.
No existe convención ni parámetro unificado. Un audio procesado por `batch_process`
→ `relabel_vtt` → `diagnose_speaker` deja artefactos en 3+ carpetas distintas.

---

## Módulos a modificar

| Módulo | Cambio |
|--------|--------|
| `audio_state.py` | Agregar `artifacts_dir: Path` derivado de `source_path` → `source_path.parent / f".{stem}"` |
| `core_analysis.py` | Crear `.{stem}/` al inicio. Pasar `artifacts_dir` a cada paso. Cache por existencia. |
| `convert_to_wav.py` | Output → `artifacts_dir/source.wav` |
| `convert_to_mono.py` | Output → `artifacts_dir/` (o skip si mono) |
| `resample_to_16k.py` | Output → `artifacts_dir/16k.wav` |
| `loudnorm.py` | Operar sobre archivo en `artifacts_dir/` |
| `enhance_audio.py` | Output → `artifacts_dir/enhanced.wav` (flat, sin subdirs ClearVoice) |
| `compress_audio.py` | Output → `artifacts_dir/compressed.m4a` |
| `speaker_recognition.py` | Temp → `artifacts_dir/tmp/`. Cache embeddings → `artifacts_dir/embeddings.npz` |
| `wav_segmenter.py` | Temp → `artifacts_dir/tmp/` |
| `write_log_file.py` | Output → `artifacts_dir/{lang}.vtt` (sin timestamp en nombre) |
| **tools/relabel_vtt.py** | Output → `artifacts_dir/relabeled_{lang}.vtt` |
| **tools/extract_unknown_speakers.py** | Clips → `artifacts_dir/unknown_speakers/` |
| **tools/diagnose_speaker.py** | Temp → `artifacts_dir/tmp/` |
| **tools/process_recordings.py** | Samples → `artifacts_dir/`, unknown clips → `artifacts_dir/unknown_speakers/` |
| **tools/batch_process.py** | Derivar `artifacts_dir` por audio, pasar a `core_analysis` y `extract_unknown_speakers` |

## Cache de mayor impacto

| Artefacto | Ahorro | Detección |
|-----------|--------|-----------|
| `enhanced.wav` | ~40 min GPU | `(artifacts_dir / "enhanced.wav").exists()` |
| `diarization.rttm` | ~2 min GPU | `(artifacts_dir / "diarization.rttm").exists()` |
| `embeddings.npz` | ~10s CPU | existe + `mtime > voices_folder mtime` |
| `16k.wav` | ~30s CPU | `(artifacts_dir / "16k.wav").exists()` |
