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

    Escribe en artifacts_dir/enhanced.wav. Si ya existe, retorna cache.
    """
    out_path = state.artifacts_dir / "enhanced.wav"
    if out_path.exists():
        return state.model_copy(update={"working_path": out_path, "is_enhanced": True})

    global _clearvoice_model
    if _clearvoice_model is None:
        _clearvoice_model = ClearVoice(
            task='speech_enhancement',
            model_names=[_MODEL_NAME],
        )

    # Directorio temporal para que ClearVoice escriba su output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = state.artifacts_dir / "_enhance_tmp"
    result_audio = _clearvoice_model(
        input_path=str(state.working_path),
        output_path=str(tmp_dir),
        online_write=False,
    )
    # Mover resultado al path canonico
    tmp_out = tmp_dir / _MODEL_NAME / state.working_path.name
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    _clearvoice_model.write(result_audio, output_path=str(tmp_out))
    tmp_out.replace(out_path)
    return state.model_copy(update={"working_path": out_path, "is_enhanced": True})
