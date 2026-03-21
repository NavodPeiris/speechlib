# guia_proceso

Reglas de decisión y comandos correctos para procesar video/audio. Orientado a agente.

---

## 0. Inspección previa obligatoria

```bash
ffprobe -v quiet -print_format json -show_streams input.mp4 | python3 -c "
import json,sys
d=json.load(sys.stdin)
for s in d['streams']:
  if s['codec_type']=='video':
    print('dims:', s['width'], 'x', s['height'])
    for sd in s.get('side_data_list',[]):
      if 'rotation' in sd: print('rotation:', sd['rotation'])
"
```

Determina el caso (A, B o C) antes de continuar.

---

## 1. Extracción de audio

```bash
ffmpeg -i input.mp4 -vn -c:a copy input.m4a
```

---

## 2. Compresión de video

### Parámetros base (todos los casos)

```bash
-c:v h264_nvenc -rc:v vbr -cq:v 28 -b:v 0 -preset p5 -tune hq
-profile:v high -level:v 4.1 -pix_fmt yuv420p -movflags +faststart -an
```

Output: `[nombre]_compress.mp4`

### Caso A — `rotation: -90`, portrait correcto en player

Frames físicos son landscape (ej. 1920×1080), metadata indica -90° → player muestra portrait.

```bash
ffmpeg -i input.mp4 -vf "scale=608:1080" \
  -c:v h264_nvenc -rc:v vbr -cq:v 28 -b:v 0 -preset p5 -tune hq \
  -profile:v high -level:v 4.1 -pix_fmt yuv420p -movflags +faststart -an \
  output_compress.mp4
```

NVENC strips rotation metadata durante encoding → output queda físicamente portrait (608×1080) sin metadata. Correcto.

### Caso B — Sin rotation metadata, visualmente rotado

**Preguntar dirección antes de ejecutar.**

Una vez confirmada la dirección:

```bash
# Si el contenido está rotado CCW (necesita girar CW para corregir):
-vf "transpose=1,scale=608:1080"

# Si el contenido está rotado CW (necesita girar CCW para corregir):
-vf "transpose=2,scale=608:1080"
```

No agregar rotation metadata al output.

### Caso C — Sin rotation, landscape correcto

```bash
ffmpeg -i input.mp4 -vf "scale=-2:1080" \
  -c:v h264_nvenc -rc:v vbr -cq:v 28 -b:v 0 -preset p5 -tune hq \
  -profile:v high -level:v 4.1 -pix_fmt yuv420p -movflags +faststart -an \
  output_compress.mp4
```

---

## 3. Merge video comprimido + audio enhanced

Sin re-encodear ningún track:

```bash
ffmpeg -i video_compress.mp4 -i audio_enhanced.m4a \
  -map 0:v -map 1:a -c:v copy -c:a copy \
  output_cleaned.mp4
```

---

## 4. Audio para transcripción

Aplicar a `_cleaned.mp4` sin re-encodear video:

```bash
ffmpeg -i input_cleaned.mp4 \
  -c:v copy -c:a aac -ac 1 -b:a 96k -ar 16000 \
  -af "loudnorm=I=-14:TP=-1.0:LRA=9" \
  output_cleaned.mp4
```

Parámetros: mono, 96k, 16kHz, loudnorm EBU R128 (I=-14, TP=-1.0, LRA=9).

---

## 5. Concatenación

### Audio (sin re-encodear)

```bash
# concat_audio.txt:
file 'audio1.m4a'
file 'audio2.m4a'

ffmpeg -f concat -safe 0 -i concat_audio.txt -c copy output_concat.m4a
```

### Video (sin re-encodear)

```bash
# concat_video.txt:
file 'video1_compress.mp4'
file 'video2_compress.mp4'

ffmpeg -f concat -safe 0 -i concat_video.txt -c copy output_concat.mp4
```

### Merge audio+video concatenados

```bash
ffmpeg -i video_concat.mp4 -i audio_concat.m4a \
  -map 0:v -map 1:a -c:v copy -c:a copy \
  output_final.mp4
```

---

## Reglas generales

- Alto máximo visual: 1080px
- GPU: h264_nvenc (RTX 2070)
- NVENC no preserva rotation metadata → diseñar el output para que sea físicamente correcto sin depender de metadata
- Nunca usar `-metadata:s:v rotate=` con NVENC — no funciona
- Separar siempre audio y video antes de comprimir
- Para transcripción: mono 16kHz 96k con loudnorm
