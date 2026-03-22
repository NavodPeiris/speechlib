"""AT: ClearVoice se invoca con online_write=False para evitar I/O sincrono durante inferencia GPU.

El resultado se escribe manualmente despues de la inferencia con write().
"""
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path


def test_enhance_audio_uses_online_write_false(tmp_path):
    """ClearVoice se invoca con online_write=False."""
    import speechlib.enhance_audio as mod
    # Reset the global model so we can mock it
    original_model = mod._clearvoice_model
    try:
        mock_model = MagicMock()
        mock_model.return_value = {"audio": "data"}
        mod._clearvoice_model = mock_model

        from speechlib.audio_state import AudioState
        state = AudioState(
            source_path=tmp_path / "test.wav",
            working_path=tmp_path / "test.wav",
        )
        (tmp_path / "test.wav").write_bytes(b"RIFF")

        result = mod.enhance_audio(state)

        # Check that __call__ was invoked with online_write=False
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
        mock_model = MagicMock()
        result_audio = {"audio": "data"}
        mock_model.return_value = result_audio
        mod._clearvoice_model = mock_model

        from speechlib.audio_state import AudioState
        state = AudioState(
            source_path=tmp_path / "test.wav",
            working_path=tmp_path / "test.wav",
        )
        (tmp_path / "test.wav").write_bytes(b"RIFF")

        mod.enhance_audio(state)

        # write() must be called with the result audio dict
        mock_model.write.assert_called_once()
        write_args = mock_model.write.call_args
        assert write_args.args[0] is result_audio or write_args.kwargs.get("result") is result_audio, (
            f"write() debe recibir el resultado de la inferencia"
        )
    finally:
        mod._clearvoice_model = original_model
