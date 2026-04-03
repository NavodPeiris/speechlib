"""AT: merge_transcript_turns fusiona segmentos consecutivos del mismo locutor
concatenando el texto.

El pipeline genera muchos segmentos cortos del mismo speaker (ej. Jolyon en
4 fragmentos de 1-14s). El output debe mostrarlos como una sola entrada con
el texto unificado, tal como aparece en el SRT de referencia.

Invariantes:
- Solo se fusionan segmentos CONSECUTIVOS del mismo speaker.
- Si un speaker distinto aparece en el medio, los bloques no se fusionan.
- El texto de los segmentos fusionados se concatena con espacio.
- Segmentos con texto vacío contribuyen solo timestamps, no texto.
"""
from speechlib.segment_merger import merge_transcript_turns


def test_merges_consecutive_same_speaker():
    """Cuatro fragmentos Jolyon consecutivos → un solo segmento."""
    segments = [
        [92.2, 106.1, "texto A", "Jolyon"],
        [106.6, 107.7, "texto B", "Jolyon"],
        [108.3, 110.2, "texto C", "Jolyon"],
        [110.7, 119.1, "texto D", "Jolyon"],
    ]
    result = merge_transcript_turns(segments, max_gap_s=2.0)
    assert len(result) == 1
    assert result[0][0] == 92.2
    assert result[0][1] == 119.1
    assert "texto A" in result[0][2]
    assert "texto D" in result[0][2]
    assert result[0][3] == "Jolyon"


def test_does_not_merge_across_speaker_change():
    """Jolyon → Agustin → Jolyon: los dos bloques Jolyon NO se fusionan."""
    segments = [
        [92.2, 106.1, "primera parte", "Jolyon"],
        [106.6, 107.7, "interrumpe", "Agustin"],
        [108.3, 119.1, "segunda parte", "Jolyon"],
    ]
    result = merge_transcript_turns(segments, max_gap_s=2.0)
    assert len(result) == 3
    jolyon_segs = [s for s in result if s[3] == "Jolyon"]
    assert len(jolyon_segs) == 2


def test_does_not_merge_large_gap():
    """Gap > umbral: NO fusionar aunque sea el mismo speaker."""
    segments = [
        [0.0, 10.0, "hola", "Jolyon"],
        [15.0, 20.0, "mundo", "Jolyon"],
    ]
    result = merge_transcript_turns(segments, max_gap_s=2.0)
    assert len(result) == 2


def test_text_concatenated_with_space():
    """El texto fusionado usa espacio como separador."""
    segments = [
        [0.0, 3.0, "Hola", "Manuel"],
        [3.4, 6.0, "mundo", "Manuel"],
    ]
    result = merge_transcript_turns(segments, max_gap_s=2.0)
    assert result[0][2] == "Hola mundo"


def test_empty_text_segment_skips_space():
    """Segmentos con texto vacío no añaden espacio extra."""
    segments = [
        [0.0, 3.0, "Hola", "Manuel"],
        [3.4, 4.0, "", "Manuel"],
        [4.2, 6.0, "mundo", "Manuel"],
    ]
    result = merge_transcript_turns(segments, max_gap_s=2.0)
    assert len(result) == 1
    assert result[0][2] == "Hola mundo"


def test_empty_list():
    assert merge_transcript_turns([]) == []


def test_single_segment():
    seg = [0.0, 5.0, "texto", "Agustin"]
    result = merge_transcript_turns([seg])
    assert result == [[0.0, 5.0, "texto", "Agustin"]]
