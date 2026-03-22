"""AT: transcribe_full_aligned asigna cada whisper segment exclusivamente
al diarization segment con mayor overlap (best-match exclusive assignment).

Bug anterior: overlap > 0 causaba que un whisper segment de 30s se asignara
a TODOS los diarization segments dentro de esa ventana → texto repetido en
múltiples lineas del output.

Invariante post-fix:
- Ningún texto de un whisper segment aparece en más de un diarization segment.
- El segment con mayor overlap recibe el texto; los demás quedan vacíos.
"""
from unittest.mock import patch, MagicMock
from speechlib.transcribe import transcribe_full_aligned


def _make_ws(start, end, text):
    ws = MagicMock()
    ws.start = start
    ws.end = end
    ws.text = text
    ws.words = []  # sin word timestamps → usa fallback segment-level
    return ws


def _make_model_and_pipeline(whisper_segs):
    mock_model = MagicMock()
    mock_model.supported_languages = ["es", "en"]

    mock_pipeline = MagicMock()
    mock_pipeline.transcribe.return_value = (iter(whisper_segs), MagicMock())
    return mock_model, mock_pipeline


def test_no_text_duplication_across_segments():
    """Whisper segment de 30s NO debe repetirse en múltiples diarization segs."""
    # Un único whisper segment que abarca toda la ventana
    ws = _make_ws(0.0, 30.0, "texto completo de treinta segundos")

    # 5 diarization segments dentro de esa ventana
    diarization = [
        [0.0, 8.0, "Francisco"],
        [8.5, 15.0, "Manuel"],
        [15.5, 20.0, "Agustin"],
        [20.5, 25.0, "Manuel"],
        [25.5, 30.0, "Francisco"],
    ]

    mock_model, mock_pipeline = _make_model_and_pipeline([ws])

    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        result = transcribe_full_aligned("audio.wav", diarization, "es", "large-v3-turbo", False)

    texts = [seg[2] for seg in result]

    # Solo un segmento debe tener el texto; el resto debe estar vacío
    non_empty = [t for t in texts if t]
    assert len(non_empty) == 1, (
        f"Se esperaba 1 segmento con texto, se encontraron {len(non_empty)}: {texts}"
    )


def test_best_overlap_wins():
    """El segmento con mayor overlap recibe el texto del whisper segment."""
    ws = _make_ws(0.0, 30.0, "texto largo")

    # Francisco (0→8s, overlap=8s) debe ganar vs Manuel (8.5→15s, overlap=6.5s)
    diarization = [
        [0.0, 8.0, "Francisco"],
        [8.5, 15.0, "Manuel"],
        [15.5, 30.0, "Agustin"],
    ]

    mock_model, mock_pipeline = _make_model_and_pipeline([ws])

    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        result = transcribe_full_aligned("audio.wav", diarization, "es", "large-v3-turbo", False)

    texts_by_speaker = {seg[3]: seg[2] for seg in result}

    # Agustin tiene el mayor overlap (14.5s) → debe ganar
    assert texts_by_speaker.get("Agustin") == "texto largo", (
        f"Agustin debería tener el texto (mayor overlap 14.5s). "
        f"Resultado: {texts_by_speaker}"
    )
    assert texts_by_speaker.get("Francisco") == "", (
        f"Francisco no debería tener texto. Resultado: {texts_by_speaker}"
    )
    assert texts_by_speaker.get("Manuel") == "", (
        f"Manuel no debería tener texto. Resultado: {texts_by_speaker}"
    )


def test_multiple_whisper_segs_distributed():
    """Múltiples whisper segments cortos se distribuyen correctamente."""
    ws_segs = [
        _make_ws(0.0, 5.0, "hola mundo"),
        _make_ws(5.0, 10.0, "como están"),
        _make_ws(10.0, 15.0, "muy bien"),
    ]

    diarization = [
        [0.0, 5.5, "Francisco"],   # ws1 best match (5s overlap)
        [5.2, 10.5, "Manuel"],     # ws2 best match (5s overlap)
        [10.0, 15.0, "Agustin"],   # ws3 best match (5s overlap)
    ]

    mock_model, mock_pipeline = _make_model_and_pipeline(ws_segs)

    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        result = transcribe_full_aligned("audio.wav", diarization, "es", "large-v3-turbo", False)

    texts_by_speaker = {seg[3]: seg[2] for seg in result}

    assert "hola mundo" in texts_by_speaker.get("Francisco", ""), (
        f"Francisco debe tener 'hola mundo'. Got: {texts_by_speaker}"
    )
    assert "como están" in texts_by_speaker.get("Manuel", ""), (
        f"Manuel debe tener 'como están'. Got: {texts_by_speaker}"
    )
    assert "muy bien" in texts_by_speaker.get("Agustin", ""), (
        f"Agustin debe tener 'muy bien'. Got: {texts_by_speaker}"
    )
