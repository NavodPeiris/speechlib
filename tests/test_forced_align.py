"""Unit tests: funciones puras de forced alignment (trellis, backtrack, merge)."""
import torch
import pytest


# ── _get_trellis ────────────────────────────────────────────────────────────

def test_get_trellis_shape():
    from speechlib.forced_align import _get_trellis
    emission = torch.randn(10, 5)  # 10 frames, 5 vocab
    tokens = [1, 2, 3]
    trellis = _get_trellis(emission, tokens, blank_id=0)
    assert trellis.shape == (11, 4)  # (T+1, N+1)


def test_get_trellis_initial_conditions():
    from speechlib.forced_align import _get_trellis
    emission = torch.randn(10, 5)
    tokens = [1, 2]
    trellis = _get_trellis(emission, tokens, blank_id=0)
    assert trellis[0, 0].item() == 0.0


# ── _backtrack ──────────────────────────────────────────────────────────────

def test_backtrack_returns_path_or_none():
    from speechlib.forced_align import _get_trellis, _backtrack
    # Create a simple emission that strongly favors tokens in sequence
    emission = torch.full((20, 4), -10.0)
    emission[:5, 0] = 0.0     # blank for first 5 frames
    emission[5:10, 1] = 0.0   # token 1 for frames 5-9
    emission[10:15, 0] = 0.0  # blank for 10-14
    emission[15:20, 2] = 0.0  # token 2 for frames 15-19
    tokens = [1, 2]
    trellis = _get_trellis(emission, tokens, blank_id=0)
    path = _backtrack(trellis, emission, tokens, blank_id=0)
    assert path is not None
    assert len(path) > 0
    # Path should contain references to both tokens
    token_indices = {p[0] for p in path}
    assert 0 in token_indices  # token_index 0 = first token
    assert 1 in token_indices  # token_index 1 = second token


# ── _merge_repeats ──────────────────────────────────────────────────────────

def test_merge_repeats_groups_consecutive():
    from speechlib.forced_align import _merge_repeats
    # Simulate path: token 0 at frames 0,1,2 and token 1 at frames 3,4
    path = [
        (0, 0, 0.9), (0, 1, 0.8), (0, 2, 0.85),
        (1, 3, 0.7), (1, 4, 0.75),
    ]
    transcript = "ab"
    segments = _merge_repeats(path, transcript)
    assert len(segments) == 2
    assert segments[0]["char"] == "a"
    assert segments[0]["start"] == 0
    assert segments[0]["end"] == 3  # exclusive end
    assert segments[1]["char"] == "b"
    assert segments[1]["start"] == 3
    assert segments[1]["end"] == 5


def test_merge_repeats_single_frame_per_token():
    from speechlib.forced_align import _merge_repeats
    path = [(0, 5, 0.9), (1, 6, 0.8)]
    transcript = "xy"
    segments = _merge_repeats(path, transcript)
    assert len(segments) == 2
    assert segments[0]["char"] == "x"
    assert segments[1]["char"] == "y"


# ── align_words con idioma no soportado ─────────────────────────────────────

def test_align_words_unsupported_language_returns_original():
    """Si el idioma no tiene modelo, retorna segmentos sin modificar."""
    import types
    from speechlib.forced_align import align_words

    word = types.SimpleNamespace(word="test", start=1.0, end=2.0)
    seg = types.SimpleNamespace(start=0.0, end=5.0, text="test", words=[word])

    result = align_words("dummy.wav", [seg], "xx", "cpu")

    # Debe retornar los mismos segmentos originales (sin crash)
    assert len(result) == 1
    assert result[0].words[0].word == "test"
    assert result[0].words[0].start == 1.0


# ── AlignedWord / AlignedSegment dataclasses ────────────────────────────────

def test_aligned_word_attributes():
    from speechlib.forced_align import AlignedWord
    w = AlignedWord(word="hola", start=1.0, end=2.0)
    assert w.word == "hola"
    assert w.start == 1.0
    assert w.end == 2.0


def test_aligned_segment_attributes():
    from speechlib.forced_align import AlignedSegment, AlignedWord
    w = AlignedWord(word="hola", start=1.0, end=2.0)
    s = AlignedSegment(start=0.0, end=5.0, text="hola", words=[w])
    assert s.start == 0.0
    assert len(s.words) == 1
