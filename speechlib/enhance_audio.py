import sys
sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")
from clearvoice import ClearVoice
from .audio_state import AudioState
from .step_timer import timed
from .kernel_profiler import timed as ktimed

_clearvoice_model = None
_MODEL_NAME = "MossFormer2_SE_48K"


@timed("enhance_audio")
@ktimed("enhance_audio")
def enhance_audio(state: AudioState) -> AudioState:
    """Aplica speech enhancement con ClearVoice MossFormer2_SE_48K.

    ClearVoice escribe en: output_dir/<ModelName>/<input_filename>
    Se crea un directorio temporal junto al archivo de trabajo.
    """
    global _clearvoice_model
    if _clearvoice_model is None:
        _clearvoice_model = ClearVoice(
            task='speech_enhancement',
            model_names=[_MODEL_NAME],
        )

    # Directorio de salida junto al archivo de trabajo
    out_dir = state.working_path.parent / (state.working_path.stem + "_enhanced_out")
    result_audio = _clearvoice_model(
        input_path=str(state.working_path),
        output_path=str(out_dir),
        online_write=False,
    )
    # Escribir resultado manualmente despues de la inferencia GPU
    out_path = out_dir / _MODEL_NAME / state.working_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _clearvoice_model.write(result_audio, output_path=str(out_path))
    return state.model_copy(update={"working_path": out_path, "is_enhanced": True})
