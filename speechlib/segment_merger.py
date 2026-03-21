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
