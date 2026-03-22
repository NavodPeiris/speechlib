"""AT: transcribe_full_aligned usa word_timestamps para asignación precisa.

Con exclusive assignment a nivel de segmento, un whisper segment de 30s puede
"robar" el texto de interjecciones cortas (0.5s "Sí", 2s "Muy bien, muy bien").
Word-level timestamps permite asignar cada palabra al speaker que la está
pronunciando según la diarización, logrando una atribución correcta incluso
para turnos cortos dentro de ventanas de transcripción largas.

Invariante: texto de interjección corta va al speaker correcto aunque compita
con un whisper segment largo que la engloba.
"""
from unittest.mock import patch, MagicMock
from speechlib.transcribe import transcribe_full_aligned


def _make_word(start, end, text):
    w = MagicMock()
    w.start = start
    w.end = end
    w.word = text
    return w


def _make_ws_with_words(start, end, words_data):
    """words_data: [(start, end, text), ...]"""
    ws = MagicMock()
    ws.start = start
    ws.end = end
    ws.text = " ".join(t for _, _, t in words_data)
    ws.words = [_make_word(s, e, t) for s, e, t in words_data]
    return ws


def _make_model_and_pipeline(whisper_segs):
    mock_model = MagicMock()
    mock_model.supported_languages = ["es", "en"]
    mock_pipeline = MagicMock()
    mock_pipeline.transcribe.return_value = (iter(whisper_segs), MagicMock())
    return mock_model, mock_pipeline


def test_short_interjection_gets_correct_speaker():
    """Manuel dice 'Muy bien' en 15-18s dentro de un whisper seg de 0-30s.

    Con word-timestamps, 'Muy bien' debe ir a Manuel, no a Francisco
    que domina el resto del whisper segment.
    """
    # Whisper segment 0-30s con palabras de distintos speakers
    words = [
        # Francisco habla 0-15s
        (0.0, 2.0, "Perfecto"),
        (2.0, 5.0, "entonces"),
        (5.0, 8.0, "lo"),
        (8.0, 10.0, "último"),
        (10.0, 13.0, "que"),
        (13.0, 15.0, "sacamos"),
        # Manuel interjección 15-18s
        (15.0, 16.0, "Muy"),
        (16.0, 18.0, "bien"),
        # Vuelve otro hablante 18-30s
        (18.0, 20.0, "El"),
        (20.0, 25.0, "resto"),
        (25.0, 30.0, "gracias"),
    ]
    ws = _make_ws_with_words(0.0, 30.0, words)

    # Diarización: Francisco 0-15s, Manuel 15-18s, otro 18-30s
    diarization = [
        [0.0, 15.0, "Francisco"],
        [15.0, 18.0, "Manuel"],
        [18.0, 30.0, "Agustin"],
    ]

    mock_model, mock_pipeline = _make_model_and_pipeline([ws])
    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        result = transcribe_full_aligned("audio.wav", diarization, "es", "large-v3-turbo", False)

    by_speaker = {seg[3]: seg[2] for seg in result}

    assert "Muy" in by_speaker.get("Manuel", "") or "bien" in by_speaker.get("Manuel", ""), (
        f"Manuel debe tener 'Muy bien'. Resultado: {by_speaker}"
    )
    # Francisco no debe tener las palabras de Manuel
    assert "Muy" not in by_speaker.get("Francisco", ""), (
        f"Francisco no debe tener 'Muy'. Resultado: {by_speaker}"
    )


def test_word_timestamps_called_in_transcribe():
    """transcribe_full_aligned debe llamar a batched.transcribe con word_timestamps=True."""
    ws = _make_ws_with_words(0.0, 5.0, [(0.0, 2.5, "hola"), (2.5, 5.0, "mundo")])
    diarization = [[0.0, 5.0, "Agustin"]]

    mock_model, mock_pipeline = _make_model_and_pipeline([ws])
    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        transcribe_full_aligned("audio.wav", diarization, "es", "large-v3-turbo", False)

    call_kwargs = mock_pipeline.transcribe.call_args
    wt = call_kwargs.kwargs.get("word_timestamps")
    assert wt is True, (
        f"Se esperaba word_timestamps=True, recibido {wt}. Call: {call_kwargs}"
    )
