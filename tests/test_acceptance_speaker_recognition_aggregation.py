"""
AT: speaker_recognition agrega embeddings antes de decidir (no vota por segmento).

Comportamiento anterior (INCORRECTO):
  - 1 segmento con score >= threshold → ese speaker gana con 1 voto
  - Resultado: falso positivo cuando el resto de segmentos no matchean a nadie

Comportamiento nuevo (CORRECTO):
  - Promediar todos los embeddings → 1 comparación con la evidencia acumulada
  - 1 segmento matching + 9 segmentos orthogonales → avg embedding ~0.11 < 0.40 → "unknown"
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torchaudio
import torch


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


# speaker_a tiene embedding [1, 0, 0]
SPEAKER_A_EMB = np.array([1.0, 0.0, 0.0])
# embedding orthogonal — cosine_sim con SPEAKER_A = 0.0
ORTHOGONAL_EMB = np.array([0.0, 1.0, 0.0])
# embedding coincidente — cosine_sim con SPEAKER_A = 1.0
MATCHING_EMB = np.array([1.0, 0.0, 0.0])


class TestSpeakerRecognitionAggregation:

    def test_single_matching_segment_among_many_orthogonal_returns_unknown(
        self, tmp_path
    ):
        """1 segmento matching + 9 orthogonales → unknown (no falso positivo).

        Avg embedding = mean([[1,0,0], [0,1,0]*9]) = [0.1, 0.9, 0]
        cosine_sim([0.1, 0.9, 0], [1, 0, 0]) = 0.1 / sqrt(0.01+0.81) ≈ 0.11 < 0.40

        Con voting antiguo: 1 voto para speaker_a, 0 para el resto → speaker_a gana (bug).
        Con agregación: avg embedding no supera threshold → "unknown" (correcto).
        """
        from speechlib.speaker_recognition import speaker_recognition

        audio = _make_wav(tmp_path / "audio.wav", duration_s=60.0)
        voices = tmp_path / "voices"
        (voices / "speaker_a").mkdir(parents=True)
        _make_wav(voices / "speaker_a" / "voice.wav")

        # 1 segmento matching + 9 orthogonales (simula Harald con 1 segmento similar a Jolyon)
        segments = [[float(i), float(i + 4), "SPEAKER_XX"] for i in range(10)]

        call_index = [0]

        def mock_inference(path):
            idx = call_index[0]
            call_index[0] += 1
            if "speaker_a" in str(path):
                return SPEAKER_A_EMB
            # Primer segmento de audio: matching; resto: orthogonales
            return MATCHING_EMB if idx == 0 else ORTHOGONAL_EMB

        with patch("speechlib.speaker_recognition._get_inference") as mock_get_inf:
            mock_get_inf.return_value.side_effect = mock_inference
            with patch("speechlib.speaker_recognition.get_embedding") as mock_ge:
                mock_ge.side_effect = lambda p: SPEAKER_A_EMB if "speaker_a" in p else None
                result = speaker_recognition(
                    str(audio), str(voices), segments, []
                )

        assert result == "unknown", (
            f"Con 1 segmento matching y 9 orthogonales, el resultado deberia ser "
            f"'unknown' (embedding agregado por debajo de threshold), pero fue '{result}'"
        )

    def test_majority_matching_segments_returns_known_speaker(self, tmp_path):
        """Cuando la mayoría de segmentos matchean un speaker → identificado correctamente."""
        from speechlib.speaker_recognition import speaker_recognition

        audio = _make_wav(tmp_path / "audio.wav", duration_s=60.0)
        voices = tmp_path / "voices"
        (voices / "speaker_a").mkdir(parents=True)
        _make_wav(voices / "speaker_a" / "voice.wav")

        segments = [[float(i), float(i + 4), "SPEAKER_XX"] for i in range(10)]

        call_index = [0]

        def mock_inference(path):
            idx = call_index[0]
            call_index[0] += 1
            if "speaker_a" in str(path):
                return SPEAKER_A_EMB
            # 8 matching, 2 orthogonales → avg ≈ [0.8, 0.2, 0] → cos_sim ≈ 0.97 > 0.40
            return MATCHING_EMB if idx < 8 else ORTHOGONAL_EMB

        with patch("speechlib.speaker_recognition._get_inference") as mock_get_inf:
            mock_get_inf.return_value.side_effect = mock_inference
            with patch("speechlib.speaker_recognition.get_embedding") as mock_ge:
                mock_ge.side_effect = lambda p: SPEAKER_A_EMB if "speaker_a" in p else None
                result = speaker_recognition(
                    str(audio), str(voices), segments, []
                )

        assert result == "speaker_a", (
            f"Con 8/10 segmentos matching, el resultado deberia ser 'speaker_a', "
            f"pero fue '{result}'"
        )

    def test_all_unknown_segments_returns_unknown(self, tmp_path):
        """Todos los segmentos orthogonales → unknown."""
        from speechlib.speaker_recognition import speaker_recognition

        audio = _make_wav(tmp_path / "audio.wav", duration_s=30.0)
        voices = tmp_path / "voices"
        (voices / "speaker_a").mkdir(parents=True)
        _make_wav(voices / "speaker_a" / "voice.wav")

        segments = [[float(i), float(i + 4), "SPEAKER_XX"] for i in range(5)]

        def mock_inference(path):
            if "speaker_a" in str(path):
                return SPEAKER_A_EMB
            return ORTHOGONAL_EMB

        with patch("speechlib.speaker_recognition._get_inference") as mock_get_inf:
            mock_get_inf.return_value.side_effect = mock_inference
            with patch("speechlib.speaker_recognition.get_embedding") as mock_ge:
                mock_ge.side_effect = lambda p: SPEAKER_A_EMB if "speaker_a" in p else None
                result = speaker_recognition(
                    str(audio), str(voices), segments, []
                )

        assert result == "unknown"
