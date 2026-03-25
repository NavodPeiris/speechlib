"""AT: ClearVoice se invoca con online_write=False para evitar I/O sincrono durante inferencia GPU.

El resultado se escribe manualmente despues de la inferencia con write().
"""
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path


def _make_state(tmp_path):
    """Crea un AudioState minimo con artifacts_dir creado."""
    from speechlib.audio_state import AudioState
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF")
    state = AudioState(source_path=wav, working_path=wav)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return state


def _fake_write_factory(state):
    """Side-effect para mock.write: crea el archivo tmp_out para que replace() funcione."""
    import speechlib.enhance_audio as mod
    def _fake_write(audio_data, output_path=None):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake_enhanced")
    return _fake_write


def test_enhance_audio_uses_online_write_false(tmp_path):
    """ClearVoice se invoca con online_write=False."""
    import speechlib.enhance_audio as mod
    original_model = mod._clearvoice_model
    try:
        state = _make_state(tmp_path)
        mock_model = MagicMock()
        mock_model.return_value = {"audio": "data"}
        mock_model.write.side_effect = _fake_write_factory(state)
        mod._clearvoice_model = mock_model

        mod.enhance_audio(state)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args
        assert call_kwargs.kwargs.get("online_write") is False, (
            f"online_write debe ser False, pero fue: {call_kwargs.kwargs.get('online_write')}"
        )
    finally:
        mod._clearvoice_model = original_model


def test_enhance_audio_writes_result_manually_after_inference(tmp_path):
    """El resultado se escribe con _clearvoice_model.write() despues de la inferencia."""
    import speechlib.enhance_audio as mod
    original_model = mod._clearvoice_model
    try:
        state = _make_state(tmp_path)
        mock_model = MagicMock()
        result_audio = {"audio": "data"}
        mock_model.return_value = result_audio
        mock_model.write.side_effect = _fake_write_factory(state)
        mod._clearvoice_model = mock_model

        mod.enhance_audio(state)

        mock_model.write.assert_called_once()
        write_args = mock_model.write.call_args
        assert write_args.args[0] is result_audio or write_args.kwargs.get("result") is result_audio, (
            f"write() debe recibir el resultado de la inferencia"
        )
    finally:
        mod._clearvoice_model = original_model
