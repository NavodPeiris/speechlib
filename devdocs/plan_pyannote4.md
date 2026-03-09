# Plan: Migrar core_analysis.py a pyannote-audio 4.x

**Contexto:** `pyannote-audio==4.0.4` cambió el modelo de pipeline, el parámetro de
autenticación y el formato de invocación respecto a 3.x.

**Metodología:** ATDD + TDD + One-Piece-Flow (devdocs/standard-atdd-tdd.md)

---

## Cambios en la API de pyannote 3.x → 4.x

| Aspecto | 3.x (actual) | 4.x (objetivo) |
|---------|--------------|----------------|
| Modelo | `"pyannote/speaker-diarization@2.1"` | `"pyannote/speaker-diarization-community-1"` |
| Auth param | `use_auth_token=ACCESS_TOKEN` | `token=ACCESS_TOKEN` |
| Invocación | `pipeline({"waveform": wf, "sample_rate": sr}, min_speakers=0, max_speakers=10)` | `pipeline(str(path))` |
| Output iter | `diarization.itertracks(yield_label=True)` → `(turn, _, speaker)` | `diarization.itertracks(yield_label=True)` → `(turn, _, speaker)` (sin cambio) |
| Audio decode | torchaudio (en `core_analysis.py`) | torchcodec interno (pyannote carga el archivo) |
| Requisito sistema | — | `ffmpeg` instalado en el sistema |

**Nota sobre el output iter:** La API `itertracks(yield_label=True)` se mantiene en
pyannote 4.x. La sintaxis `output.speaker_diarization` es alternativa pero opcional.
El loop existente no necesita cambio.

---

## Slice 1 — Actualizar inicialización del pipeline

### Acceptance Test (RED hasta que producción esté lista)

```python
# tests/test_acceptance_pyannote4.py

from unittest.mock import patch, MagicMock
from pathlib import Path

def test_pipeline_initialized_with_community1_model(tmp_path):
    """
    core_analysis llama a Pipeline.from_pretrained con el modelo community-1
    y el parámetro token= (no use_auth_token=).
    """
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 36)  # minimal WAV header stub

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end   = 1.0

    mock_pipeline = MagicMock()
    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [(mock_segment, None, "SPEAKER_00")]
    mock_pipeline.return_value = mock_diarization

    with patch("speechlib.core_analysis.Pipeline") as mock_cls, \
         patch("speechlib.core_analysis.convert_to_wav",  side_effect=lambda s: s), \
         patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s), \
         patch("speechlib.core_analysis.re_encode",       side_effect=lambda s: s), \
         patch("speechlib.core_analysis.wav_file_segmentation", return_value=[]), \
         patch("speechlib.core_analysis.write_log_file"):

        mock_cls.from_pretrained.return_value = mock_pipeline

        state_mock = MagicMock()
        state_mock.working_path = wav

        from speechlib.core_analysis import core_analysis
        core_analysis(str(wav), None, "logs", "en", "tiny", "TOKEN", "whisper")

        mock_cls.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-community-1",
            token="TOKEN",
        )
```

### Unit tests (RED)

```python
# tests/test_core_analysis_pyannote4.py

from unittest.mock import patch, MagicMock, call
from pathlib import Path

def _make_mock_pipeline(mock_cls, segments):
    """Helper: configura mock_cls.from_pretrained para devolver pipeline con segments."""
    mock_pipeline = MagicMock()
    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = segments
    mock_pipeline.return_value = mock_diarization
    mock_cls.from_pretrained.return_value = mock_pipeline
    return mock_pipeline, mock_diarization


def test_from_pretrained_uses_community1_model(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.touch()

    with patch("speechlib.core_analysis.Pipeline") as mock_cls, \
         patch("speechlib.core_analysis.convert_to_wav",  side_effect=lambda s: s), \
         patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s), \
         patch("speechlib.core_analysis.re_encode",       side_effect=lambda s: s), \
         patch("speechlib.core_analysis.wav_file_segmentation", return_value=[]), \
         patch("speechlib.core_analysis.write_log_file"):

        mock_seg = MagicMock(); mock_seg.start = 0.0; mock_seg.end = 1.0
        _make_mock_pipeline(mock_cls, [(mock_seg, None, "SPEAKER_00")])

        from speechlib.core_analysis import core_analysis
        core_analysis(str(wav), None, "logs", "en", "tiny", "MY_TOKEN", "whisper")

        args, kwargs = mock_cls.from_pretrained.call_args
        assert args[0] == "pyannote/speaker-diarization-community-1"
        assert "use_auth_token" not in kwargs
        assert kwargs.get("token") == "MY_TOKEN"


def test_pipeline_called_with_file_path(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.touch()

    with patch("speechlib.core_analysis.Pipeline") as mock_cls, \
         patch("speechlib.core_analysis.convert_to_wav",  side_effect=lambda s: s), \
         patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s), \
         patch("speechlib.core_analysis.re_encode",       side_effect=lambda s: s), \
         patch("speechlib.core_analysis.wav_file_segmentation", return_value=[]), \
         patch("speechlib.core_analysis.write_log_file"):

        mock_seg = MagicMock(); mock_seg.start = 0.0; mock_seg.end = 1.0
        mock_pipeline, _ = _make_mock_pipeline(mock_cls, [(mock_seg, None, "SPEAKER_00")])

        from speechlib.core_analysis import core_analysis
        core_analysis(str(wav), None, "logs", "en", "tiny", "TOKEN", "whisper")

        # pipeline debe llamarse con str del path, no con dict waveform
        call_args = mock_pipeline.call_args
        assert isinstance(call_args[0][0], str)
        assert "waveform" not in str(call_args)


def test_itertracks_still_yields_3tuple(tmp_path):
    """El loop de diarización sigue funcionando con (turn, _, speaker)."""
    wav = tmp_path / "audio.wav"
    wav.touch()

    with patch("speechlib.core_analysis.Pipeline") as mock_cls, \
         patch("speechlib.core_analysis.convert_to_wav",  side_effect=lambda s: s), \
         patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s), \
         patch("speechlib.core_analysis.re_encode",       side_effect=lambda s: s), \
         patch("speechlib.core_analysis.wav_file_segmentation", return_value=[[0.0, 1.0, "hello"]]), \
         patch("speechlib.core_analysis.write_log_file"):

        mock_seg = MagicMock(); mock_seg.start = 0.0; mock_seg.end = 1.0
        _make_mock_pipeline(mock_cls, [(mock_seg, None, "SPEAKER_00")])

        from speechlib.core_analysis import core_analysis
        result = core_analysis(str(wav), None, "logs", "en", "tiny", "TOKEN", "whisper")

        assert any(seg[3] == "SPEAKER_00" for seg in result)
```

### Producción — `speechlib/core_analysis.py`

Reemplazar el bloque de inicialización y llamada al pipeline:

```python
# ANTES
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                use_auth_token=ACCESS_TOKEN)
...
waveform, sample_rate = torchaudio.load(str(state.working_path))
...
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, min_speakers=0, max_speakers=10)

# DESPUÉS
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                token=ACCESS_TOKEN)
...
diarization = pipeline(str(state.working_path))
```

Eliminar la importación `torchaudio` si no se usa en otro lugar del archivo
(actualmente solo se usa para `torchaudio.load` antes del pipeline).

El loop `for turn, _, speaker in diarization.itertracks(yield_label=True):` no cambia.

### Prerrequisito de sistema

```bash
# ffmpeg debe estar instalado (requerido por torchcodec en pyannote 4.x)
# Windows:
winget install ffmpeg
# o scoop install ffmpeg
```

### Verificar GREEN

```bash
conda activate wx3
pytest tests/test_core_analysis_pyannote4.py tests/test_acceptance_pyannote4.py -v
```

### Commit

```
feat: migrate core_analysis to pyannote-audio 4.x API

- Pipeline model: speaker-diarization@2.1 -> community-1
- Auth param: use_auth_token= -> token=
- Invocation: waveform dict -> file path string
- Remove torchaudio.load from core_analysis (torchcodec handles it internally)
```

---

## Files touched

| File | Cambio |
|------|--------|
| `speechlib/core_analysis.py` | Modelo, param auth, invocación del pipeline |
| `tests/test_core_analysis_pyannote4.py` | Unit tests nuevos |
| `tests/test_acceptance_pyannote4.py` | AT nuevo |

---

## Prerequisitos antes de implementar

1. `ffmpeg` instalado en el sistema
2. Env wx3: `pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128`
3. `pip install pyannote-audio==4.0.4`
4. Aceptar condiciones del modelo `pyannote/speaker-diarization-community-1` en HuggingFace
