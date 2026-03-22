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


def absorb_micro_segments(common: list, threshold: float = 0.3) -> list:
    """Absorbe segmentos < threshold en su vecino más largo, sin importar speaker.

    El vecino que absorbe extiende su timestamp para cubrir el rango del micro.
    El speaker del micro desaparece — el vecino mantiene su speaker.
    """
    if not common:
        return []

    segments = [list(s) for s in common]
    absorbed = set()

    for i, seg in enumerate(segments):
        duration = seg[1] - seg[0]
        if duration >= threshold:
            continue

        # Buscar vecino más largo (izquierdo o derecho, no absorbido)
        left_dur = 0.0
        left_idx = None
        for j in range(i - 1, -1, -1):
            if j not in absorbed:
                left_idx = j
                left_dur = segments[j][1] - segments[j][0]
                break

        right_dur = 0.0
        right_idx = None
        for j in range(i + 1, len(segments)):
            if j not in absorbed:
                right_idx = j
                right_dur = segments[j][1] - segments[j][0]
                break

        if left_idx is None and right_idx is None:
            continue  # solo queda este segmento, preservar

        # Absorber en el vecino más largo
        if left_dur >= right_dur and left_idx is not None:
            target = left_idx
        elif right_idx is not None:
            target = right_idx
        else:
            target = left_idx

        # Extender timestamps del vecino
        segments[target][0] = min(segments[target][0], seg[0])
        segments[target][1] = max(segments[target][1], seg[1])
        absorbed.add(i)

    return [seg for i, seg in enumerate(segments) if i not in absorbed]


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
