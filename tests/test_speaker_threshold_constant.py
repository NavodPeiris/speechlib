"""AT: SPEAKER_SIMILARITY_THRESHOLD constante única en speaker_recognition."""
import numpy as np


def test_constant_value_is_040():
    from speechlib.speaker_recognition import SPEAKER_SIMILARITY_THRESHOLD
    assert SPEAKER_SIMILARITY_THRESHOLD == 0.40


def test_find_best_speaker_default_uses_constant():
    """find_best_speaker sin threshold explícito debe usar SPEAKER_SIMILARITY_THRESHOLD."""
    from speechlib.speaker_recognition import find_best_speaker, SPEAKER_SIMILARITY_THRESHOLD

    # embedding idéntico al de "alice" → score=1.0, siempre supera threshold
    emb = np.array([1.0, 0.0, 0.0])
    speaker_embs = {"alice": np.array([1.0, 0.0, 0.0])}

    result = find_best_speaker(emb, speaker_embs)
    assert result == "alice"


def test_find_best_speaker_below_threshold_returns_unknown():
    from speechlib.speaker_recognition import find_best_speaker, SPEAKER_SIMILARITY_THRESHOLD

    # embedding ortogonal → score≈0, bajo threshold
    emb = np.array([1.0, 0.0, 0.0])
    speaker_embs = {"alice": np.array([0.0, 1.0, 0.0])}

    result = find_best_speaker(emb, speaker_embs)
    assert result == "unknown"


def test_relabel_vtt_imports_threshold_from_core():
    """relabel_vtt no debe definir DEFAULT_THRESHOLD localmente; debe importarlo del core."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] /
           "speechlib" / "tools" / "relabel_vtt.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # No debe haber asignación literal: DEFAULT_THRESHOLD = 0.40
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_THRESHOLD":
                    if isinstance(node.value, ast.Constant):
                        raise AssertionError("DEFAULT_THRESHOLD definido como literal en relabel_vtt.py")


def test_diagnose_speaker_imports_threshold_from_core():
    """diagnose_speaker no debe definir THRESHOLD localmente."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] /
           "speechlib" / "tools" / "diagnose_speaker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "THRESHOLD":
                    if isinstance(node.value, ast.Constant):
                        raise AssertionError("THRESHOLD definido como literal en diagnose_speaker.py")
