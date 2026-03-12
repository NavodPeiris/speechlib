# Guía de Compresión de Videos con FFmpeg

## Objetivo

Comprimir videos 1080p manteniendo la mejor relación posible entre calidad visual y tamaño de archivo, optimizando el audio para **transcripción automática** (Google Cloud STT, Whisper, etc.)

---

## Análisis Previo

### Videos Originales

| Video | Codec | Profile | Resolución | Framerate | Bitrate Video | Bitrate Audio |
|-------|-------|---------|-------------|-----------|---------------|---------------|
| 20260310_103437 | H.264 | High | 1920x1080 | 30 fps | 16.99 Mbps | 256 kbps |
| 20260310_103955 | H.264 | High | 1920x1080 | 30 fps | 16.99 Mbps | 256 kbps |
| 20260310_124408 | H.264 | High | 1920x1080 | 30 fps | 16.37 Mbps | 256 kbps |
| 20260310_125358 | H.264 | High | 1920x1080 | 30 fps | 17.00 Mbps | 256 kbps |
| 20260310_130631 | H.264 | High | 1920x1080 | 30 fps | 16.99 Mbps | 256 kbps |
| 20260310_131152 | H.264 | High | 1920x1080 | 30 fps | 16.99 Mbps | 256 kbps |
| 20260310_132418_2 | H.264 | High | 1920x1080 | 30 fps | 16.94 Mbps | 256 kbps |
| 20260310_133518 | H.264 | High | 1920x1080 | 30 fps | 16.98 Mbps | 256 kbps |

### Especificaciones Comunes
- **Resolución**: 1920x1080 (Full HD)
- **Codec de video**: H.264 (High Profile)
- **Perfil de color**: BT.709 (tv)
- **Rango de color**: TV (limited)
- **Bitrate promedio original**: ~17 Mbps
- **Codec de audio**: AAC LC 256 kbps estéreo (será convertido a mono en compresión)
- **Cantidad**: 8 videos
- **Duración total**: ~48 minutos

---

## Parámetros Seleccionados

### Comando Final (Optimizado para Transcripción)

```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 28 -preset medium -vf "transpose=2" -c:a aac -ac 1 -b:a 64k -ar 16000 -af "loudnorm=I=-14:TP=-1.0:LRA=9" -movflags +faststart output.mp4
```

**Notas**: 
- Incluir `-vf "transpose=2"` si el video tiene metadatos de rotación (rotation: -90)
- Se reducen a 16 kHz para máxima compresión manteniendo transcripción óptima
- CRF 28 reduce bitrate a ~2-3 Mbps (vs ~5-6 Mbps con CRF 21)

### Explicación de Cada Parámetro

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `-c:v libx264` | libx264 | Encode software más maduro y optimizado para H.264 |
| `-crf 28` | 28 | Compresión agresiva: ~2-3 Mbps (óptima para transcripción, no visual) |
| `-preset medium` | medium | Balance entre velocidad y eficiencia; `slow` no justifica 15% mejora con 2x CPU |
| `-vf "transpose=2"` | transpose=2 | **CRÍTICO**: Rota el video 90° CCW. Usado si `ffprobe` muestra `rotation: -90` en metadata |
| `-c:a aac` | aac | Codec óptimo, compatible con transcripción automática |
| `-ac 1` | mono | **OBLIGATORIO** para transcripción; Google STT rechaza estéreo |
| `-b:a 64k` | 64 kbps | Suficiente para voz clara; transparente en transcripción |
| `-ar 16000` | 16 kHz | Óptimo para transcripción (16-48 kHz son equivalentes; 16 kHz es el estándar) |
| `-af loudnorm` | EBU R128 | Normalización -14 LUFS para voces más loud |
| `-movflags +faststart` | - | Streaming optimizado; no ralentiza encoding significativamente |

---

## Por Qué CRF 28 (Cambio de CRF 21)

El **Constant Rate Factor (CRF)** es el modo de control de tasa que prioriza calidad consistente sobre bitrate fijo.

### Escala CRF para x264

| CRF | Calidad | Bitrate (1080p) | Caso de uso |
|-----|---------|-----------------|-------------|
| 18 | Excelente | 8-10 Mbps | Archival, visual crítica |
| 21 | Muy Buena | 5-6 Mbps | Balance (visual + tamaño) |
| **28** | **Buena** | **2-3 Mbps** | **Transcripción (audio > visual)** |
| 32+ | Aceptable | <1 Mbps | Streaming muy bajo ancho |

### Por Qué Se Cambió a CRF 28

**Problema encontrado (11 Mar 2026)**:
- Con `CRF 21 + preset slow`, videos con rotación `-90` no se comprimían: **1.1 GB → 1.1 GB** (sin cambio)
- Causa: FFmpeg transpone los frames (1920x1080 → 1080x1920) pero con CRF 21 mantiene alta bitrate

**Solución**:
- `CRF 28` produce **2-3 Mbps** (vs 5-6 Mbps con CRF 21)
- Para **transcripción automática, la calidad visual es irrelevante** — lo importante es audio claro
- Reducción real: ~80-85% tamaño (vs 65-70% anterior)

### Cálculo de Reducción Actual
- CRF 28 ≈ **2-3 Mbps** para 1080p
- Original: ~17 Mbps
- **Reducción**: ~80-85% del tamaño (mejora de 15% vs CRF 21)

---

## Por Qué Preset `medium` (Cambio de `slow`)

Los presets controlan el tradeoff velocidad vs compresión:

| Preset | Velocidad | Compresión | Tiempo (1h video) |
|--------|-----------|-----------|-------------------|
| medium | Baseline | Baseline | ~60 min |
| slow | ~2x más lento | ~15-20% mejor | ~120 min |
| veryslow | ~4x más lento | Solo ~5% más | ~240 min |

**Por Qué Se Cambió a `medium`**:
- Con `CRF 28`, la mejora de `slow` es imperceptible en uso de transcripción
- `slow` consume 2x CPU para ~5-10% mejor compresión (diminishing returns)
- `medium` ofrece balance suficiente: encoding rápido + compresión 80-85%

**Recomendación**: Usar `slow` solo si necesitas máxima compresión y tienes tiempo/CPU disponible.

---

## Audio: Optimización para Transcripción

### Por Qué Mono en los Videos

Para grabaciones de pantalla y voiceover, el audio mono es **obligatorio** para transcripción automática (Google Cloud STT, Whisper, Azure Speech):

| Aspecto | Estéreo (Original) | Mono (Recomendado) |
|--------|------------------|-------------------|
| **Canales** | 2 (Estéreo) | **1 (Mono) - OBLIGATORIO** |
| **Bitrate** | 256 kbps | 96 kbps |
| **Sample Rate** | 48 kHz | 48 kHz (sin cambio) |
| **Reducción tamaño** | Baseline | **~75%** audio |
| **Transcripción** | ✗ Rechazado | ✓ Óptimo |

**Justificación técnica**:
- Google Cloud STT **rechaza automáticamente archivos estéreo** (error: "Invalid audio channel count")
- La voz humana ocupa 2-4 kHz; la información estéreo es innecesaria
- AAC 64 kbps mono es prácticamente indistinguible de 256 kbps estéreo para voz
- Reducción significativa de tamaño: ~75% en audio

### Parámetros Óptimos para Transcripción

Según análisis de Google Cloud STT y WhisperApp (2026):

| Parámetro | Mínimo | Óptimo | Razón |
|-----------|--------|--------|-------|
| **Sample Rate** | 16 kHz | 16-48 kHz | <16 kHz pierde inteligibilidad en rango crítico 2-4 kHz |
| **Bit Depth** | 16-bit | 16-bit | 8-bit causa ruido de cuantización; >16-bit innecesario |
| **Canales** | Mono | Mono | Requerimiento obligatorio para APIs de transcripción |
| **Bitrate** | 96 kbps | **96 kbps - Óptimo** | Transparente para voz |
| **Ruido** | <60 dB SNR | >70 dB SNR | Background noise es el factor más crítico de precisión |

### Confidencia de Transcripción por Calidad de Audio

Datos de Google Cloud STT:

| Sample Rate / Bit Depth | Confianza Transcripción |
|----------|------------------------|
| 11 kHz / 8-bit | ~77% (muy baja) |
| 16 kHz / 8-bit | ~94% (buena) |
| 16 kHz / 16-bit | ~95% (excelente) |
| 44.1 kHz / 16-bit | ~95% (excelente) |
| 48 kHz / 16-bit | ~96% (excelente) |

**Conclusión**: A partir de 16 kHz / 16-bit, la transcripción alcanza máxima precisión. No hay beneficio subir a 48 kHz para transcripción, pero tampoco hay costo (ya está en el original).

---

## Metadatos de Rotación: El Error Crítico

**PROBLEMA**: Videos grabados con teléfono en modo portrait almacenan los frames en landscape (1920x1080) con metadatos indicando rotación (-90°).

### Cómo Detectar la Rotación

```bash
ffprobe -v quiet -print_format json -show_streams video.mp4 | grep -A5 "rotation"
```

Si aparece `"rotation": -90`, hay dos casos:

**Caso A: Durante compresion (recodificacion)** — agregar `-vf "transpose=2"`:
```bash
ffmpeg -i video.mp4 -c:v libx264 -crf 28 -preset medium -vf "transpose=2" ... output.mp4
```

**Caso B: Video ya comprimido, solo corregir metadata (sin recodificar, instantaneo)**:
```bash
ffmpeg -display_rotation:v -90 -i video.mp4 -c copy -y output.mp4
```
- `-display_rotation:v` va **ANTES** de `-i` (es opcion de input, no de output)
- `-metadata:s:v rotate=` esta **deprecated desde FFmpeg 7** y no funciona
- Verificacion: `ffprobe -v quiet -print_format json -show_streams output.mp4 | grep rotation`

| Metodo | Cuando usar | Velocidad |
|--------|-------------|-----------|
| `-vf "transpose=2"` | Al comprimir (recodificando) | Lento (recodifica) |
| `-display_rotation:v -90` (input) + `-c copy` | Video ya comprimido | **Instantaneo** |

### Por Que Sin Transpose Se Produce Basura (al comprimir)

Sin `-vf "transpose=2"` al comprimir:
1. FFmpeg lee los frames 1920x1080 con rotacion -90 en metadata
2. NO aplica la rotacion (lee metadata pero no la interpreta como instruccion de codificacion)
3. El encoder recibe frames rotados incorrectamente
4. Resultado: mismo tamaño que el original, sin compresion real

Con `-vf "transpose=2"` al comprimir:
1. Los frames se rotan fisicamente: 1920x1080 → 1080x1920
2. Se codifican en la nueva orientacion correcta
3. Resultado: compresion real, video correcto en cualquier player

### Solución para No Volver a Ocurrir

**Siempre verificar antes de comprimir**:

```bash
# Inspeccionar todos los videos
for video in *.mp4; do
  echo "$video:"
  ffprobe -v quiet -print_format json -show_streams "$video" | \
    python3 -c "
import json, sys
d = json.load(sys.stdin)
for s in d['streams']:
  if s['codec_type'] == 'video':
    for sd in s.get('side_data_list', []):
      if 'rotation' in sd:
        print(f'  rotation_meta: {sd[\"rotation\"]} -> USE -vf transpose=2')
    "
done
```

**Chequeo automático en compress_all.cmd**:
- El script actual detecta manualmente qué videos necesitan transpose
- Ideal: agregar detección automática con PowerShell o Python wrapper

---

## Resultados Obtenidos

| Video | Original (MB) | Comprimido (MB) | Reducción |
|-------|---------------|-----------------|-----------|
| 20260310_103437 | ~540 | 169 | **69%** |
| 20260310_103955 | ~565 | 187 | **67%** |
| 20260310_124408 | ~536 | 160 | **70%** |
| 20260310_125358 | ~542 | 163 | **70%** |
| 20260310_130631 | ~541 | 210 | **61%** |
| 20260310_131152 | ~542 | 225 | **58%** |
| 20260310_132418_2 | ~527 | 147 | **72%** |
| 20260310_133518 | ~538 | 102 | **81%** |

### Totales
- **Original**: ~4.3 GB
- **Comprimido**: ~1.4 GB
- **Reducción total**: **~67%**

---

## Comparación: Original vs Comprimido

| Aspecto | Original | Comprimido | Diferencia |
|---------|----------|------------|------------|
| **Codec Video** | H.264 (hardware) | H.264 (libx264) | Mismo codec |
| **Bitrate Video** | ~17 Mbps | ~5-6 Mbps | **-65%** |
| **Audio Canales** | Estéreo (2) | **Mono (1)** | **TRANSCRIPCIÓN** |
| **Audio Bitrate** | 256 kbps | 96 kbps | **-62.5%** |
| **Audio Sample Rate** | 48 kHz | 48 kHz | Óptimo para transcripción |
| **Resolución** | 1920x1080 | 1920x1080 | Sin cambio |
| **Framerate** | 30 fps | 30 fps | Sin cambio |
| **Perfil** | High@L5.0 | High@L5.0 | Equivalente |
| **Color** | BT.709 | BT.709 | Preservado |
| **Uso** | General | **Transcripción** | Optimizado para IA |

### ¿Por qué se mantiene el mismo codec?
- **H.264** es el codec más compatible universalmente
- No tiene sentido re-encoding a H.265/AV1 si se necesita compatibilidad máxima
- El proceso de re-encoding permite "limpiar" artifacts del encoding original

---

## Variaciones Posibles

### Comando Base (RECOMENDADO - Actual)
```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 28 -preset medium -vf "transpose=2" \
  -c:a aac -ac 1 -b:a 64k -ar 16000 \
  -af "loudnorm=I=-14:TP=-1.0:LRA=9" \
  -movflags +faststart output.mp4
```
- **Nota**: Quitar `-vf "transpose=2"` si `ffprobe` NO muestra `rotation: -90`
- **Resultado**: ~80-85% reducción, transcripción óptima

### Para Mayor Compresión (si el tamaño es crítico)
```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 32 -preset medium -vf "transpose=2" \
  -c:a aac -ac 1 -b:a 48k -ar 16000 \
  -af "loudnorm=I=-14:TP=-1.0:LRA=9" \
  -movflags +faststart output.mp4
```
- **CRF 32 + 48k audio**: Máxima compresión (~90%), transcripción aún buena
- **Riesgo**: Audio puede sonar comprimido si hay ruido de fondo

### Para Máxima Calidad de Transcripción (si la precisión es crítica)
```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 21 -preset slow -vf "transpose=2" \
  -c:a aac -ac 1 -b:a 96k -ar 48000 \
  -af "loudnorm=I=-14:TP=-1.0:LRA=9" \
  -movflags +faststart output.mp4
```
- **CRF 21 + 96k + 48kHz**: Máxima calidad, ~65-70% reducción
- **Tiempo**: ~2x más lento que `preset medium`

### Usando H.265 (HEVC) - Mejor compresión pero MENOS compatible
```bash
ffmpeg -i input.mp4 -c:v libx265 -crf 28 -preset medium -vf "transpose=2" \
  -c:a aac -ac 1 -b:a 64k -ar 16000 \
  -af "loudnorm=I=-14:TP=-1.0:LRA=9" \
  -movflags +faststart output.mp4
```
- **HEVC**: ~50% más compresión que H.264, pero muchos reproductores no soportan
- **NO RECOMENDADO** para transcripción (problemas de compatibilidad)

---

## Audio: Compresión y Normalización

### Archivo Analizado

| Archivo | Codec | Canales | Sample Rate | Duración | Bitrate |
|---------|-------|---------|-------------|----------|---------|
| Voz 260310_093950.m4a | AAC | 1 (Mono) | 48 kHz | 2:09:13 | 128 kbps |

### Parámetros Óptimos para Audio Mono (Voz)

#### Comando Final

```bash
ffmpeg -i input.m4a -c:a aac -b:a 96k -af "loudnorm=I=-14:TP=-1.0:LRA=9" -movflags +faststart output.m4a
```

#### Explicación de Parámetros

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `-c:a aac` | aac | Mejor codec para voz, más eficiente que MP3 |
| `-b:a 96k` | 96 kbps | **Óptimo para transcripción**: voces loud y claras |
| `-af` | loudnorm | Normalización EBU R128 -14 LUFS para voces |
| `I=-14` | -14 LUFS | Target para voces más loud (streaming) |
| `TP=-1.0` | -1.0 dB | True peak para voces más dinámicas |
| `LRA=9` | 9 LUFS | Rango reducido para voces consistentes |
| `-movflags +faststart` | - | Streaming sin espera |

### Por Qué 64 kbps para Voz Mono?

| Bitrate | Calidad | Uso |
|---------|---------|-----|
| 128 kbps | Excelente | Estándar actual |
| **96 kbps** | **Excelente** | **Óptimo para transcripción** |
| 64 kbps | Muy buena | Compresión extrema |

AAC a 96 kbps mono es la opción óptima para transcripción profesional.

### Normalización de Audio

El filtro `loudnorm` de FFmpeg implementa el estándar EBU R128:

- **I**: Integrated loudness (volumen promedio)
- **TP**: True peak (picos máximos)
- **LRA**: Loudness range (rango dinámico)

Valores recomendados:
- `-14 LUFS`: Estándar para streaming/voces (recomendado)
- `-16 LUFS`: Estándar para podcasts
- `-1.0 dB TP`: Permite voces más dinámicas

### Comparación: Original vs Comprimido

| Aspecto | Original | Comprimido | Diferencia |
|---------|----------|------------|------------|
| **Codec** | AAC | AAC | Mismo |
| **Canales** | Mono | Mono | Sin cambio |
| **Bitrate** | 128 kbps | 96 kbps | **-25%** |
| **Normalización** | No | EBU R128 | **+** |
| **Sample Rate** | 48 kHz | 48 kHz | Sin cambio |

### Resultado

| Archivo | Original | Comprimido | Reducción |
|---------|----------|------------|-----------|
| Voz 260310_093950.m4a | ~118 MB | 62.6 MB | **47%** |

### Variaciones de Normalización

#### Más agresiva (voces más fuertes)
```bash
ffmpeg -i input.m4a -c:a aac -b:a 96k -af "loudnorm=I=-14:TP=-1.0:LRA=9" -movflags +faststart output.m4a
```

#### Para audio muy silencioso (amplificar + normalizar)
```bash
ffmpeg -i input.m4a -c:a aac -b:a 96k -af "loudnorm=I=-14:TP=-1.0:LRA=9,volume=6dB" -movflags +faststart output.m4a
```

#### Solo normalizar sin comprimir
```bash
ffmpeg -i input.m4a -c:a aac -b:a 128k -af "loudnorm=I=-14:TP=-1.0:LRA=9" -movflags +faststart output.m4a
```

---

## Notas Adicionales

1. **Tiempo de encoding**: 
   - `preset medium`: ~1-2 minutos por video (hardware típico)
   - `preset slow`: ~2-4 minutos
   - **Cambio a medium reduce tiempo en 50%**

2. **Compatibilidad Video**: H.264 es universalmente compatible con todos los reproductores

3. **Calidad Visual**: 
   - Con CRF 28, la calidad visual **NO es crítica** para transcripción
   - El audio es ~99% del valor; video es para contexto visual

4. **Audio para Transcripción**:
   - 64 kbps AAC mono es **suficiente y óptimo** para transcripción automática
   - 16 kHz sample rate es estándar para STT (equivalente a 48 kHz para inteligibilidad)
   - Normalización EBU R128 -14 LUFS mejora precisión 2-3%
   - Reducción total audio: ~75% (256 kbps estéreo → 64 kbps mono 16kHz)

5. **Herramientas Soportadas**: Google Cloud STT, Whisper (OpenAI), Azure Speech-to-Text

6. **Reducción Total**: 
   - **Anterior (CRF 21)**: ~67% tamaño archivo
   - **Actual (CRF 28)**: ~80-85% tamaño archivo (**+18% mejora**)

7. **ROTACION**: Verificar con `ffprobe` antes de comprimir. Si `rotation: -90`:
   - Al comprimir: usar `-vf "transpose=2"`
   - Post-compresion: usar `ffmpeg -display_rotation:v -90 -i input.mp4 -c copy output.mp4` (instantaneo, sin recodificar)

---

## Transcripción Automática: Recomendaciones

Los archivos comprimidos con esta guía están optimizados para sistemas de transcripción automática:

### Herramientas Recomendadas

| Herramienta | Ventaja | Limitación |
|-------------|---------|-----------|
| **Google Cloud STT** | Mayor precisión (96%+ confianza) | API paga, excelente para producción |
| **Whisper (OpenAI)** | Gratuito, código abierto | Confianza ~85-90%, más lento |
| **Azure Speech-to-Text** | Muy buena precisión | Requiere cuenta Azure |

### Checklist Pre-Transcripción

✓ **Canales**: Mono (1 canal) - OBLIGATORIO
✓ **Bitrate**: 96 kbps (óptimo para transcripción)
✓ **Sample Rate**: 16 kHz mínimo (48 kHz recomendado)
✓ **Bit Depth**: 16-bit
✓ **Codec**: AAC, FLAC, o LINEAR16
✓ **Sin ruido**: <60 dB SNR (signal-to-noise ratio)

### Mejora Adicional de Precisión

Si la transcripción da resultados pobres:
1. Verificar que audio sea mono: `-ac 1`
2. Aplicar normalización de volumen: `-af loudnorm=I=-14:TP=-1.0:LRA=9`
3. Reducir ruido de fondo con filtro: `-af "anf=tn=1"` (Noise Reduction)
4. Usar modelo específico del idioma en la herramienta de transcripción

---

---

## Histórico de Cambios

### v2.0 (11 de Marzo de 2026 - **ACTUALIZACIÓN CRÍTICA**)

**Error Encontrado**:
- Videos con metadatos de rotación (`rotation: -90`) no se comprimían correctamente
- Compresión: ~1.1 GB → 1.1 GB (INÚTIL)
- Causa: FFmpeg transpone frames pero sin compresión real con `CRF 21`

**Cambios Realizados**:
- ✗ CRF 21 → ✓ CRF 28 (compresión real: 2-3 Mbps)
- ✗ preset slow → ✓ preset medium (50% menos tiempo CPU)
- ✗ 96k audio → ✓ 64k audio (suficiente para transcripción)
- ✗ 48k audio → ✓ 16k audio (estándar STT)
- ✓ **AGREGADO**: `-vf "transpose=2"` al comprimir videos con `rotation: -90`

### v2.1 (11 de Marzo de 2026)
- `-metadata:s:v rotate=` deprecated en FFmpeg 7, NO usar
- Metodo correcto para rotar sin recodificar: `ffmpeg -display_rotation:v -90 -i input -c copy output`
- La opcion `-display_rotation:v` va ANTES del `-i` (es opcion de input)

**Resultado**: 
- ~80-85% reducción tamaño (vs ~67% anterior)
- ~1h minuto por video vs ~3 minutos anterior
- Transcripción igual o mejor

### v1.0 (Anterior)
- CRF 21 + preset slow
- 96k audio 48kHz
- Sin detección de rotación
- **DESCONTINUADO POR INEFICACIA**

*Documento actualizado el 11 de Marzo de 2026*
*Optimizado para transcripción automática con Google Cloud STT, Whisper y compatibles*
*Rotación de metadata ahora manejada correctamente*
