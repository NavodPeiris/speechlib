"""
AT: find_best_speaker retorna 'unknown' cuando el score cae bajo el umbral.

Sin umbral, un speaker desconocido como Patricio Renner se asignaría
incorrectamente al más cercano en la librería (e.g. Agustin).
"""

import numpy as np
import pytest


def test_high_confidence_returns_known_speaker():
    """Score > threshold → retorna nombre conocido."""
    from speechlib.speaker_recognition import find_best_speaker

    embeddings = {"Agustin": np.array([[1.0, 0.0, 0.0]])}
    test_emb = np.array([[1.0, 0.0, 0.0]])  # idéntico → score = 1.0
    result = find_best_speaker(test_emb, embeddings, threshold=0.75)
    assert result == "Agustin"


def test_low_confidence_returns_unknown():
    """Score < threshold → retorna 'unknown' aunque haya candidatos."""
    from speechlib.speaker_recognition import find_best_speaker

    embeddings = {
        "Agustin":  np.array([[1.0, 0.0, 0.0]]),
        "Francisco": np.array([[0.0, 1.0, 0.0]]),
    }
    # Voz no parecida a ninguno → score ~0.0
    test_emb = np.array([[0.0, 0.0, 1.0]])
    result = find_best_speaker(test_emb, embeddings, threshold=0.75)
    assert result == "unknown"


def test_score_exactly_at_threshold_returns_speaker():
    """Score == threshold → se considera suficiente (límite inclusive)."""
    from speechlib.speaker_recognition import find_best_speaker

    # Crear embedding cuyo cosine similarity sea exactamente 0.75
    # cos(θ) = a·b / (|a||b|); con a=[1,0,0] y b=[0.75, sqrt(1-0.75²), 0]
    import math
    val = 0.75
    perp = math.sqrt(1 - val**2)
    embeddings = {"Agustin": np.array([[1.0, 0.0, 0.0]])}
    test_emb = np.array([[val, perp, 0.0]])
    result = find_best_speaker(test_emb, embeddings, threshold=0.75)
    assert result == "Agustin"


def test_default_threshold_is_075():
    """Sin pasar threshold, el default es 0.75."""
    from speechlib.speaker_recognition import find_best_speaker

    embeddings = {"Agustin": np.array([[1.0, 0.0, 0.0]])}
    test_emb = np.array([[0.0, 0.0, 1.0]])  # score ~0
    result = find_best_speaker(test_emb, embeddings)
    assert result == "unknown"


def test_empty_library_returns_unknown():
    """Sin voces en la librería → siempre unknown."""
    from speechlib.speaker_recognition import find_best_speaker

    result = find_best_speaker(np.array([[1.0, 0.0, 0.0]]), {})
    assert result == "unknown"
