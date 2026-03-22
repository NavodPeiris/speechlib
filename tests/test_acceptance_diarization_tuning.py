"""AT: El pipeline de diarizacion usa min_duration_off para reducir fragmentacion.

Configurar min_duration_off >= 0.5s en los parametros del pipeline de segmentacion
reduce la over-segmentation de pyannote, produciendo menos segmentos y por tanto
menos trabajo de transcripcion.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def clear_diarization_cache():
    from speechlib.core_analysis import _get_diarization_pipeline
    _get_diarization_pipeline.cache_clear()
    yield
    _get_diarization_pipeline.cache_clear()


def test_diarization_uses_configured_min_duration_off():
    """El pipeline de diarizacion usa min_duration_off >= 0.5s para reducir fragmentacion."""
    from speechlib.core_analysis import _get_diarization_pipeline

    mock_pipeline = MagicMock()

    with patch("speechlib.core_analysis.Pipeline.from_pretrained", return_value=mock_pipeline):
        result = _get_diarization_pipeline("TOKEN")

    # Verify that the segmentation parameters were set with min_duration_off
    mock_pipeline.instantiate.assert_called_once()
    call_args = mock_pipeline.instantiate.call_args
    params = call_args.args[0] if call_args.args else call_args.kwargs.get("parameters", {})

    # Check that segmentation min_duration_off is set >= 0.5
    seg_params = params.get("segmentation", {})
    assert "min_duration_off" in seg_params, (
        f"min_duration_off no configurado en segmentation params: {params}"
    )
    assert seg_params["min_duration_off"] >= 0.5, (
        f"min_duration_off={seg_params['min_duration_off']}, esperado >= 0.5"
    )
