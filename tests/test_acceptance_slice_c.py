"""AT: BatchedInferencePipeline es usado en faster-whisper (Slice C).

La transcripción con model_type='faster-whisper' debe usar BatchedInferencePipeline
con batch_size=16 en lugar de model.transcribe() directamente.
"""
import pytest
from unittest.mock import patch, MagicMock, call
from speechlib.transcribe import transcribe


@pytest.fixture(autouse=True)
def clear_cache():
    from speechlib.transcribe import _get_faster_whisper_model
    _get_faster_whisper_model.cache_clear()
    yield
    _get_faster_whisper_model.cache_clear()


def _make_mock_whisper_model():
    mock_model = MagicMock()
    mock_model.supported_languages = ["en", "es", "fr"]
    return mock_model


def _make_mock_batched_pipeline(text="hello world"):
    mock_batched = MagicMock()
    mock_seg = MagicMock()
    mock_seg.text = text
    mock_batched.transcribe.return_value = ([mock_seg], MagicMock())
    return mock_batched


def test_batched_pipeline_is_used_for_faster_whisper(tmp_path):
    """AT: faster-whisper usa BatchedInferencePipeline, no model.transcribe() directamente."""
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")

    mock_model = _make_mock_whisper_model()
    mock_batched = _make_mock_batched_pipeline()

    with (
        patch("speechlib.transcribe.WhisperModel", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched) as MockBatched,
    ):
        result = transcribe(str(wav), "en", "base", "faster-whisper", False, None, None, None)

    MockBatched.assert_called_once_with(model=mock_model)
    mock_batched.transcribe.assert_called_once()
    assert isinstance(result, str)


def test_batched_transcribe_called_with_batch_size_16(tmp_path):
    """AT: batch_size=16 se pasa a batched.transcribe()."""
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")

    mock_model = _make_mock_whisper_model()
    mock_batched = _make_mock_batched_pipeline()

    with (
        patch("speechlib.transcribe.WhisperModel", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched),
    ):
        transcribe(str(wav), "en", "base", "faster-whisper", False, None, None, None)

    call_kwargs = mock_batched.transcribe.call_args
    assert call_kwargs.kwargs.get("batch_size") == 16 or (
        len(call_kwargs.args) > 1 and call_kwargs.args[1] == 16
    ), f"batch_size=16 no encontrado en llamada: {call_kwargs}"


def test_batched_returns_text_string(tmp_path):
    """AT: transcribe() retorna str con el texto concatenado de los segmentos."""
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")

    mock_model = _make_mock_whisper_model()
    mock_batched = _make_mock_batched_pipeline(text=" hello world ")

    with (
        patch("speechlib.transcribe.WhisperModel", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched),
    ):
        result = transcribe(str(wav), "en", "base", "faster-whisper", False, None, None, None)

    assert "hello world" in result
    assert isinstance(result, str)


def test_model_cache_still_works_with_batched(tmp_path):
    """AT: el LRU cache de WhisperModel sigue funcionando con BatchedInferencePipeline."""
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")

    mock_model = _make_mock_whisper_model()
    mock_batched = _make_mock_batched_pipeline()

    with (
        patch("speechlib.transcribe.WhisperModel", return_value=mock_model) as MockModel,
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched),
    ):
        transcribe(str(wav), "en", "base", "faster-whisper", False, None, None, None)
        transcribe(str(wav), "en", "base", "faster-whisper", False, None, None, None)

    assert MockModel.call_count == 1, (
        f"WhisperModel construido {MockModel.call_count} veces — cache no funciona"
    )
