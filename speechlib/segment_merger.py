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
