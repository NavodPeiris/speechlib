# 🎯 Plan de Implementación: Audio Enhancement (LUFS + ClearVoice)

**Objetivo:** Mejorar precisión de transcripción de reuniones en +30% mediante:
1. Normalización LUFS (-23 LUFS broadcast standard)
2. Audio Enhancement con ClearVoice (opcional)

**Principios:**
- ✅ Backward compatibility 100%
- ✅ Modificaciones mínimas al core
- ✅ API idiomática Python
- ✅ End-to-end en cada slice
- ✅ Fallbacks robustos

---

## 📋 Slices Incrementales

### SLICE 0: Análisis y Preparación (~1 hora)

**Deliverables:**
- Análisis de código actual
- Checklist de cambios necesarios
- Plan de testing

**Tasks:**

1. **Revisar arquitectura actual**
   - [ ] Mapear todos los módulos en `speechlib/`
   - [ ] Identificar flujo de preprocessing: `convert_to_wav` → `convert_to_mono` → `re_encode` → diarización
   - [ ] Ubicar punto de inserción ideal para normalización (DESPUÉS de re_encode)

2. **Verificar dependencias disponibles**
   - [ ] Confirmar que `torch` y `torchaudio` ya existen (para LUFS)
   - [ ] Revisar `pyproject.toml` y `requirements.txt`
   - [ ] Listar dependencias opcionales (librosa, ffmpeg-python)

3. **Crear checklist de cambios**
   - [ ] `normalize_lufs.py` (NUEVO)
   - [ ] `PreProcessor` class (MODIFICAR - 1 línea)
   - [ ] `core_analysis.py` (MODIFICAR - 2-3 líneas)
   - [ ] `__init__.py` (OPCIONAL - si exponer la clase)
   - [ ] `examples/preprocess.py` (ACTUALIZAR)
   - [ ] `README.md` (DOCUMENTAR)
   - [ ] Tests unitarios (NUEVO)

**Definition of Done:**
- ✅ Documento de análisis completado
- ✅ Checklist actualizado
- ✅ Confirmación de que NO requiere cambios en dependencies

---

### SLICE 1: Módulo LUFS Normalization (~2-3 horas)

**Deliverables:**
- Archivo `speechlib/normalize_lufs.py` completo y funcional
- Unit tests
- Documentación interna

**Tasks:**

1. **Crear `speechlib/normalize_lufs.py`**

```python
# speechlib/normalize_lufs.py
"""
LUFS (Loudness Units relative to Full Scale) normalization.
Especially useful for meeting transcriptions with variable audio levels.

Target: -23 LUFS (ITU R.128 broadcast standard)
"""

import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

TARGET_LUFS = -23.0
SILENCE_THRESHOLD = -69.0


def measure_lufs_torch(waveform: torch.Tensor, sr: int) -> float:
    """
    Measure loudness in LUFS using PyTorch (fast, CPU/GPU compatible).
    
    Args:
        waveform: Audio tensor [channels, samples] or [samples]
        sr: Sample rate in Hz
        
    Returns:
        Loudness in LUFS, or -70.0 if silent
        
    Example:
        >>> waveform, sr = torchaudio.load("audio.wav")
        >>> lufs = measure_lufs_torch(waveform, sr)
        >>> print(f"Current loudness: {lufs:.1f} LUFS")
    """
    # Handle mono/stereo
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    
    # Compute RMS
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-10)
    
    # Convert to dBFS
    dbfs = 20 * torch.log10(rms + 1e-10)
    
    # Approximate LUFS from dBFS
    # (Simplified; real ITU R.128 is more complex with frequency weighting)
    lufs = dbfs.item() - 0.691
    
    # Return silence marker if too quiet or NaN
    if lufs < -70 or not np.isfinite(lufs):
        return -70.0
    
    return float(lufs)


def normalize_lufs(
    src: Union[str, Path],
    dst: Union[str, Path],
    target_lufs: float = TARGET_LUFS,
    use_gpu: bool = False
) -> bool:
    """
    Normalize audio file to target LUFS.
    
    Args:
        src: Input audio file path
        dst: Output audio file path
        target_lufs: Target loudness (-23.0 for broadcast)
        use_gpu: Use GPU if available (torch only)
        
    Returns:
        True if successful, False if fallback (copied file as-is)
        
    Behavior:
        - Silent audio (<= -69 LUFS): copies without processing
        - On error: falls back to file copy
        - Applies soft clipping to prevent distortion
        
    Example:
        >>> from pathlib import Path
        >>> src = Path("noisy_meeting.wav")
        >>> dst = Path("normalized_meeting.wav")
        >>> success = normalize_lufs(src, dst)
        >>> if success:
        ...     print("Audio normalized successfully")
    """
    import shutil
    
    src = Path(src)
    dst = Path(dst)
    
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(src))
        
        # Optionally move to GPU
        if use_gpu and torch.cuda.is_available():
            waveform = waveform.to("cuda")
        
        # Measure current loudness
        current_lufs = measure_lufs_torch(waveform, sr)
        
        # If silent, just copy
        if current_lufs <= SILENCE_THRESHOLD:
            logger.debug(f"Audio is silent ({current_lufs:.1f} LUFS) - copying without change")
            shutil.copy2(src, dst)
            return True
        
        # Calculate required gain
        gain_db = target_lufs - current_lufs
        gain_db = np.clip(gain_db, -30.0, 30.0)  # Clamp unreasonable gains
        
        # Apply gain in dB
        gain_linear = 10 ** (gain_db / 20.0)
        normalized = waveform * gain_linear
        
        # Soft clipping to prevent distortion
        max_val = normalized.abs().max()
        if max_val > 0.99:
            normalized = normalized * (0.99 / max_val)
            logger.debug(f"Applied soft clipping (peak was {max_val:.2f})")
        
        # Move back to CPU before saving
        if use_gpu and torch.cuda.is_available():
            normalized = normalized.cpu()
        
        # Save normalized audio
        torchaudio.save(str(dst), normalized, sr)
        
        logger.info(
            f"Normalized: {current_lufs:.1f} LUFS → {target_lufs:.1f} LUFS "
            f"(gain: {gain_db:+.1f} dB)"
        )
        return True
    
    except Exception as e:
        # Fallback: copy original file
        logger.warning(f"Normalization failed ({e}) - falling back to copy")
        try:
            shutil.copy2(src, dst)
            return False
        except Exception as copy_err:
            logger.error(f"Fallback copy also failed: {copy_err}")
            return False


# Backward compatibility wrapper
def measure_lufs(wav: Union[str, Path]) -> float:
    """Measure LUFS of a WAV file. Backward compatible."""
    wav = Path(wav)
    waveform, sr = torchaudio.load(str(wav))
    return measure_lufs_torch(waveform, sr)