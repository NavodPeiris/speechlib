import re


def _is_sentence_end(text: str) -> bool:
    return bool(re.search(r'[.!?;]["\'\)]*\s*$', text))


def _is_strong_pause(text: str) -> bool:
    return bool(re.search(r'[,:]["\'\)]*\s*$', text))


def group_by_sentences(
    segments: list,
    max_chars: int = 80,
    max_duration_s: float = 10.0,
) -> list:
    """Agrupa segmentos consecutivos del mismo speaker hasta límite de sentencia.

    Opera sobre segmentos [start, end, text, speaker].
    Hace flush del buffer cuando:
    - Cambia el speaker
    - El texto acumulado termina en sentencia (. ! ? ;)
    - Supera max_chars o max_duration_s Y termina en pausa fuerte (, :)
    """
    if not segments:
        return []

    result = []
    buf_start, buf_end, buf_text, buf_speaker = segments[0]
    buf_start = float(buf_start)
    buf_end = float(buf_end)
    buf_text = buf_text or ""

    def _flush():
        result.append([buf_start, buf_end, buf_text.strip(), buf_speaker])

    for curr in segments[1:]:
        c_start, c_end, c_text, c_speaker = curr
        c_text = c_text or ""

        if c_speaker != buf_speaker:
            _flush()
            buf_start, buf_end, buf_text, buf_speaker = float(c_start), float(c_end), c_text, c_speaker
            continue

        # Same speaker: decide whether to flush before accumulating
        if _is_sentence_end(buf_text):
            _flush()
            buf_start, buf_end, buf_text, buf_speaker = float(c_start), float(c_end), c_text, c_speaker
            continue

        merged_text = (buf_text + " " + c_text).strip() if buf_text else c_text
        merged_dur = float(c_end) - buf_start
        over_limit = len(merged_text) > max_chars or merged_dur > max_duration_s
        if over_limit and _is_strong_pause(buf_text):
            _flush()
            buf_start, buf_end, buf_text, buf_speaker = float(c_start), float(c_end), c_text, c_speaker
        else:
            buf_end = float(c_end)
            buf_text = merged_text

    _flush()
    return result


def group_by_speaker(segments: list) -> list:
    """Fusiona todos los segmentos consecutivos del mismo speaker (sin límite de gap)."""
    return merge_transcript_turns(segments, max_gap_s=float("inf"))


def merge_transcript_turns(segments: list, max_gap_s: float = 2.0) -> list:
    """Fusiona segmentos consecutivos del mismo locutor tras la transcripción.

    Opera sobre segmentos con formato [start, end, text, speaker].
    Concatena el texto de los turnos fusionados con espacio.
    Solo fusiona segmentos CONSECUTIVOS — un speaker distinto en el medio
    interrumpe la cadena.
    """
    if not segments:
        return []
    result = [list(segments[0])]
    for curr in segments[1:]:
        prev = result[-1]
        gap = curr[0] - prev[1]
        if prev[3] == curr[3] and gap < max_gap_s:
            prev[1] = curr[1]
            if curr[2]:
                prev[2] = (prev[2] + " " + curr[2]).strip() if prev[2] else curr[2]
        else:
            result.append(list(curr))
    return result


def merge_short_turns(common: list, max_gap_s: float = 0.5) -> list:
    """Fusiona turnos consecutivos del mismo locutor si el gap es < max_gap_s."""
    if not common:
        return []
    result = [list(common[0])]
    for curr in common[1:]:
        prev = result[-1]
        gap = curr[0] - prev[1]
        if prev[2] == curr[2] and gap < max_gap_s:
            prev[1] = curr[1]
        else:
            result.append(list(curr))
    return result
