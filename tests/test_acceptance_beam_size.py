"""AT: transcribe_full_aligned usa beam_size=1 para maximizar throughput.

beam_size=5 multiplica el trabajo del decoder por 5 sin mejora significativa
de calidad para audio de buena calidad. beam_size=1 (greedy) es suficiente.
"""
from unittest.mock import patch, MagicMock
from speechlib.transcribe import transcribe_full_aligned


def _make_mock_model():
    mock = MagicMock()
    mock.supported_languages = ["es", "en"]
    return mock


def _make_mock_pipeline(segments=None):
    mock = MagicMock()
    if segments is None:
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 3.0
        seg.text = "hola mundo"
        segments = [seg]
    mock.transcribe.return_value = (segments, MagicMock())
    return mock


def test_transcribe_full_aligned_uses_beam_size_1():
    """transcribe_full_aligned debe llamar a batched.transcribe con beam_size=1."""
    mock_model = _make_mock_model()
    mock_pipeline = _make_mock_pipeline()

    diarization_segs = [[0.0, 3.0, "Agustin"]]

    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        transcribe_full_aligned("audio.wav", diarization_segs, "es", "large-v3-turbo", False)

    call_kwargs = mock_pipeline.transcribe.call_args
    beam = call_kwargs.kwargs.get("beam_size") or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
    assert beam == 1, f"beam_size esperado 1, recibido {beam}. Call: {call_kwargs}"
