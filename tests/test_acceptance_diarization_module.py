"""
Slice 6 AT: get_diarization_pipeline es importable desde speechlib.diarization
y mantiene el cache entre llamadas.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_get_diarization_pipeline_importable_from_diarization_module():
    from speechlib.diarization import get_diarization_pipeline
    assert callable(get_diarization_pipeline)


def test_get_diarization_pipeline_returns_callable_pipeline():
    from speechlib.diarization import get_diarization_pipeline

    mock_pipeline = MagicMock()
    with patch("speechlib.diarization.Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipeline
        get_diarization_pipeline.cache_clear()
        result = get_diarization_pipeline("fake_token")
        get_diarization_pipeline.cache_clear()

    assert callable(result)


def test_get_diarization_pipeline_caches_same_object_for_same_token():
    from speechlib.diarization import get_diarization_pipeline

    mock_pipeline = MagicMock()
    with patch("speechlib.diarization.Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipeline
        get_diarization_pipeline.cache_clear()
        p1 = get_diarization_pipeline("TOKEN_X")
        p2 = get_diarization_pipeline("TOKEN_X")
        get_diarization_pipeline.cache_clear()

    assert p1 is p2
    assert MockPipeline.from_pretrained.call_count == 1


def test_core_analysis_still_exports_get_diarization_pipeline():
    """core_analysis._get_diarization_pipeline sigue disponible para mocks existentes."""
    from speechlib.core_analysis import _get_diarization_pipeline
    assert callable(_get_diarization_pipeline)
