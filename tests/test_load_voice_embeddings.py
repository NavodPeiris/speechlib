"""AT: speaker_recognition.load_voice_embeddings / load_avg_voice_embeddings."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
import torchaudio
import torch


def _make_wav(path: Path):
    wf = torch.zeros(1, 16000)
    torchaudio.save(str(path), wf, 16000, bits_per_sample=16)


def test_skips_underscore_prefixed_dirs(tmp_path):
    from speechlib.speaker_recognition import load_voice_embeddings

    (tmp_path / "_hidden").mkdir()
    _make_wav(tmp_path / "_hidden" / "a.wav")
    (tmp_path / "alice").mkdir()
    _make_wav(tmp_path / "alice" / "a.wav")

    fake_emb = np.array([[1.0, 0.0]])
    with patch("speechlib.speaker_recognition.get_embedding", return_value=fake_emb):
        result = load_voice_embeddings(tmp_path)

    assert "_hidden" not in result
    assert "alice" in result


def test_returns_per_file_embedding_list(tmp_path):
    from speechlib.speaker_recognition import load_voice_embeddings

    (tmp_path / "bob").mkdir()
    _make_wav(tmp_path / "bob" / "1.wav")
    _make_wav(tmp_path / "bob" / "2.wav")

    call_count = [0]
    def fake_emb(path):
        call_count[0] += 1
        return np.array([[float(call_count[0]), 0.0]])

    with patch("speechlib.speaker_recognition.get_embedding", side_effect=fake_emb):
        result = load_voice_embeddings(tmp_path)

    assert "bob" in result
    assert len(result["bob"]) == 2


def test_avg_computes_mean(tmp_path):
    from speechlib.speaker_recognition import load_avg_voice_embeddings

    (tmp_path / "carol").mkdir()
    _make_wav(tmp_path / "carol" / "1.wav")
    _make_wav(tmp_path / "carol" / "2.wav")

    embs = [np.array([[1.0, 0.0]]), np.array([[3.0, 0.0]])]
    emb_iter = iter(embs)
    with patch("speechlib.speaker_recognition.get_embedding", side_effect=lambda _: next(emb_iter)):
        result = load_avg_voice_embeddings(tmp_path)

    expected = np.array([[2.0, 0.0]])
    np.testing.assert_allclose(result["carol"], expected)


def test_speaker_without_wavs_excluded(tmp_path):
    from speechlib.speaker_recognition import load_voice_embeddings

    (tmp_path / "empty").mkdir()  # no wav files

    with patch("speechlib.speaker_recognition.get_embedding") as mock_emb:
        result = load_voice_embeddings(tmp_path)

    assert "empty" not in result
    mock_emb.assert_not_called()


def test_enhanced_loads_from_enhanced_subdir(tmp_path):
    """enhanced=True carga WAVs de _enhanced/ en vez de raíz."""
    from speechlib.speaker_recognition import load_voice_embeddings

    alice = tmp_path / "alice"
    alice.mkdir()
    _make_wav(alice / "raw.wav")
    enhanced = alice / "_enhanced"
    enhanced.mkdir()
    _make_wav(enhanced / "enh.wav")

    raw_emb = np.array([[1.0, 0.0]])
    enh_emb = np.array([[0.0, 1.0]])

    def fake_emb(path):
        return enh_emb if "_enhanced" in str(path) else raw_emb

    with patch("speechlib.speaker_recognition.get_embedding", side_effect=fake_emb):
        result = load_voice_embeddings(tmp_path, enhanced=True)

    assert "alice" in result
    assert len(result["alice"]) == 1
    np.testing.assert_array_equal(result["alice"][0], enh_emb)


def test_enhanced_falls_back_to_root_when_no_enhanced_subdir(tmp_path):
    """enhanced=True sin _enhanced/ → carga WAVs raíz como fallback."""
    from speechlib.speaker_recognition import load_voice_embeddings

    alice = tmp_path / "alice"
    alice.mkdir()
    _make_wav(alice / "raw.wav")

    raw_emb = np.array([[1.0, 0.0]])

    with patch("speechlib.speaker_recognition.get_embedding", return_value=raw_emb):
        result = load_voice_embeddings(tmp_path, enhanced=True)

    assert "alice" in result
    assert len(result["alice"]) == 1
    np.testing.assert_array_equal(result["alice"][0], raw_emb)


def test_avg_enhanced_computes_mean_from_enhanced(tmp_path):
    """load_avg_voice_embeddings(enhanced=True) promedia embeddings de _enhanced/."""
    from speechlib.speaker_recognition import load_avg_voice_embeddings

    carol = tmp_path / "carol"
    carol.mkdir()
    _make_wav(carol / "raw.wav")
    enhanced = carol / "_enhanced"
    enhanced.mkdir()
    _make_wav(enhanced / "1.wav")
    _make_wav(enhanced / "2.wav")

    raw_emb = np.array([[10.0, 0.0]])
    enh_embs = iter([np.array([[1.0, 0.0]]), np.array([[3.0, 0.0]])])

    def fake_emb(path):
        return next(enh_embs) if "_enhanced" in str(path) else raw_emb

    with patch("speechlib.speaker_recognition.get_embedding", side_effect=fake_emb):
        result = load_avg_voice_embeddings(tmp_path, enhanced=True)

    expected = np.array([[2.0, 0.0]])
    np.testing.assert_allclose(result["carol"], expected)
