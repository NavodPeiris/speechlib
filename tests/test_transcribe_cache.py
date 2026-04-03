"""Tests: WhisperModel es construido una sola vez por configuración única."""
import pytest
from unittest.mock import patch, MagicMock
from speechlib.transcribe import _get_faster_whisper_model


@pytest.fixture(autouse=True)
def clear_cache():
    _get_faster_whisper_model.cache_clear()
    yield
    _get_faster_whisper_model.cache_clear()


def test_cache_returns_same_instance():
    """Mismos parámetros → mismo objeto en memoria."""
    with patch("speechlib.transcribe.WhisperModel") as MockModel:
        MockModel.return_value = MagicMock()
        inst1 = _get_faster_whisper_model("base", "cpu", "float32")
        inst2 = _get_faster_whisper_model("base", "cpu", "float32")
        assert inst1 is inst2


def test_model_constructed_once_for_n_calls():
    """WhisperModel.__init__ se llama exactamente 1 vez para 10 llamadas idénticas."""
    with patch("speechlib.transcribe.WhisperModel") as MockModel:
        MockModel.return_value = MagicMock()
        for _ in range(10):
            _get_faster_whisper_model("large-v3", "cpu", "int8")
        assert MockModel.call_count == 1


def test_cache_hit_count():
    """cache_info().hits sube con cada llamada repetida."""
    with patch("speechlib.transcribe.WhisperModel") as MockModel:
        MockModel.return_value = MagicMock()
        for _ in range(5):
            _get_faster_whisper_model("small", "cpu", "float32")
        assert _get_faster_whisper_model.cache_info().hits == 4


def test_different_params_create_different_instances():
    """Parámetros distintos → instancias distintas (ambas construidas)."""
    with patch("speechlib.transcribe.WhisperModel") as MockModel:
        MockModel.side_effect = [MagicMock(), MagicMock()]
        inst_a = _get_faster_whisper_model("base", "cpu", "float32")
        inst_b = _get_faster_whisper_model("medium", "cpu", "float32")
        assert inst_a is not inst_b
        assert MockModel.call_count == 2
