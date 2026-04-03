"""
AT: enroll_speaker — agregar nuevo speaker a la librería de voces
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from conftest import make_tone_wav


class TestDurationFiltering:
    """Filtrar clips por duración mínima"""

    def test_select_clips_filters_by_min_duration(self, tmp_path):
        """Dado: 5 clips WAV (2 de 1s, 3 de 3s)
        Con: min_duration_s=2.0
        Resultado: solo los 3 clips de >=2s son copiados
        """
        from speechlib.tools.enroll_speaker import select_clips

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        (clips_dir / "clip1.wav").write_bytes(b"RIFF")
        (clips_dir / "clip2.wav").write_bytes(b"RIFF")

        for i in range(3):
            make_tone_wav(clips_dir / f"long{i}.wav", duration_s=3.0)

        with patch(
            "speechlib.tools.enroll_speaker.get_audio_duration"
        ) as mock_duration:

            def fake_duration(path):
                name = path.name
                if "long" in name:
                    return 3.0
                return 1.0

            mock_duration.side_effect = fake_duration

            result = select_clips(
                clips_dir=clips_dir,
                speaker_name="NuevoSpeaker",
                voices_dir=voices_dir,
                min_duration_s=2.0,
                max_clips=5,
            )

        speaker_dir = voices_dir / "NuevoSpeaker"
        copied_files = list(speaker_dir.glob("*.wav"))

        assert speaker_dir.exists()
        assert len(copied_files) == 3
        assert all("long" in f.name for f in copied_files)


class TestOutlierRejection:
    """Descartar outliers por distancia al centroide"""

    def test_reject_outliers_removes_orthogonal_clip(self, tmp_path):
        """Dado: 6 clips (5 con embedding similar, 1 ortogonal)
        Con: max_clips=6
        Resultado: el clip ortogonal NO se copia
        """
        from speechlib.tools.enroll_speaker import _reject_outliers

        clips = [tmp_path / f"clip{i}.wav" for i in range(6)]

        import numpy as np

        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0]),
            np.array([1.0, 0.05, 0.0]),
            np.array([0.95, 0.05, 0.0]),
            np.array([0.85, 0.15, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]

        result_clips, result_embeddings = _reject_outliers(clips, embeddings)

        assert len(result_clips) == 5
        assert tmp_path / "clip5.wav" not in result_clips

    def test_reject_outliers_keeps_all_when_std_zero(self, tmp_path):
        """Caso borde: todos los clips son idénticos (std=0)
        Resultado: ninguno descartado
        """
        from speechlib.tools.enroll_speaker import _reject_outliers

        clips = [tmp_path / f"clip{i}.wav" for i in range(3)]

        import numpy as np

        embeddings = [np.array([1.0, 0.0, 0.0])] * 3

        result_clips, result_embeddings = _reject_outliers(clips, embeddings)

        assert len(result_clips) == 3


class TestDiversitySelection:
    """Selección greedy por diversidad de embeddings"""

    def test_select_diverse_prefers_different_embeddings(self, tmp_path):
        """Dado: 6 clips (4 similares, 2 distintos)
        Con: max_clips=3
        Resultado: al menos 1 de los distintos está incluido
        """
        from speechlib.tools.enroll_speaker import _select_diverse

        clips = [tmp_path / f"clip{i}.wav" for i in range(6)]

        import numpy as np

        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.95, 0.05, 0.0]),
            np.array([0.9, 0.1, 0.0]),
            np.array([0.85, 0.15, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        result = _select_diverse(clips, embeddings, max_clips=3)

        assert len(result) == 3
        result_names = {r.name for r in result}
        assert "clip4.wav" in result_names or "clip5.wav" in result_names


class TestVoicesLibraryEnrollment:
    """Enrollment en librería de voces con merge"""

    def test_merge_existing_speaker_continues_numeration(self, tmp_path):
        """Dado: voices/SpeakerA/ ya existe con 2 clips
        Con: 3 nuevos clips candidatos
        Resultado: 5 clips total, numeración continua (segment_03.wav, segment_04.wav, segment_05.wav)
        """
        from speechlib.tools.enroll_speaker import enroll_speaker

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        for i in range(3):
            make_tone_wav(clips_dir / f"clip{i}.wav", duration_s=3.0)

        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        speaker_dir = voices_dir / "SpeakerA"
        speaker_dir.mkdir()
        make_tone_wav(speaker_dir / "segment_01.wav", duration_s=3.0)
        make_tone_wav(speaker_dir / "segment_02.wav", duration_s=3.0)

        import numpy as np

        def fake_embedding(path):
            return np.array([0.5, 0.5, 0.0])

        with patch(
            "speechlib.tools.enroll_speaker.get_audio_duration", return_value=3.0
        ):
            with patch(
                "speechlib.tools.enroll_speaker.get_embedding",
                side_effect=fake_embedding,
            ):
                result = enroll_speaker(
                    clips_dir=clips_dir,
                    speaker_name="SpeakerA",
                    voices_dir=voices_dir,
                    min_duration_s=2.0,
                    max_clips=5,
                )

        all_clips = list(speaker_dir.glob("segment_*.wav"))
        assert len(all_clips) == 5
        names = {c.name for c in all_clips}
        assert "segment_03.wav" in names
        assert "segment_04.wav" in names
        assert "segment_05.wav" in names

    def test_new_speaker_creates_directory(self, tmp_path):
        """Dado: speaker nuevo (directorio no existe)
        Resultado: directorio creado + clips copiados con numeración
        """
        from speechlib.tools.enroll_speaker import enroll_speaker

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        for i in range(2):
            make_tone_wav(clips_dir / f"clip{i}.wav", duration_s=3.0)

        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        import numpy as np

        def fake_embedding(path):
            return np.array([0.5, 0.5, 0.0])

        with patch(
            "speechlib.tools.enroll_speaker.get_audio_duration", return_value=3.0
        ):
            with patch(
                "speechlib.tools.enroll_speaker.get_embedding",
                side_effect=fake_embedding,
            ):
                result = enroll_speaker(
                    clips_dir=clips_dir,
                    speaker_name="NuevoSpeaker",
                    voices_dir=voices_dir,
                    min_duration_s=2.0,
                    max_clips=5,
                )

        speaker_dir = voices_dir / "NuevoSpeaker"
        assert speaker_dir.exists()
        clips = list(speaker_dir.glob("*.wav"))
        assert len(clips) == 2
        names = {c.name for c in clips}
        assert "segment_01.wav" in names
        assert "segment_02.wav" in names
