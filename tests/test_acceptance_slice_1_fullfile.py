"""AT: transcripcion usa una sola llamada a batched.transcribe para todo el audio.

En lugar de N llamadas (una por segmento de diarizacion), debe hacerse UNA sola llamada
con el audio completo. Los resultados se mapean a segmentos por timestamp overlap.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def clear_cache():
    from speechlib.transcribe import _get_faster_whisper_model
    _get_faster_whisper_model.cache_clear()
    yield
    _get_faster_whisper_model.cache_clear()


def _make_whisper_segments():
    """Simula segmentos whisper con timestamps: 3 chunks cubriendo 0-10s."""
    seg1 = MagicMock(); seg1.start = 0.0; seg1.end = 3.5; seg1.text = "Hello world"
    seg2 = MagicMock(); seg2.start = 3.5; seg2.end = 7.0; seg2.text = " how are you"
    seg3 = MagicMock(); seg3.start = 7.0; seg3.end = 10.0; seg3.text = " goodbye"
    return [seg1, seg2, seg3]


def _mock_batched_pipeline(whisper_segments):
    mock_batched = MagicMock()
    mock_batched.transcribe.return_value = (iter(whisper_segments), MagicMock())
    return mock_batched


def _mock_whisper_model():
    mock_model = MagicMock()
    mock_model.supported_languages = ["en", "es", "fr"]
    return mock_model


class TestBatchedTranscribeCalledOnce:
    """Con N segmentos de diarizacion, batched.transcribe se llama UNA sola vez."""

    def test_batched_transcribe_called_once_not_per_segment(self):
        """Con 5 segmentos, transcribe_full_aligned llama batched.transcribe UNA vez."""
        from speechlib.transcribe import transcribe_full_aligned

        diarization_segments = [
            [0.0, 2.0, "SPEAKER_00"],
            [2.0, 4.0, "SPEAKER_01"],
            [4.0, 6.0, "SPEAKER_00"],
            [6.0, 8.0, "SPEAKER_01"],
            [8.0, 10.0, "SPEAKER_00"],
        ]
        whisper_segs = _make_whisper_segments()
        mock_model = _mock_whisper_model()
        mock_batched = _mock_batched_pipeline(whisper_segs)

        with (
            patch("speechlib.transcribe.WhisperModel", return_value=mock_model),
            patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched),
        ):
            transcribe_full_aligned(
                "audio.wav", diarization_segments, "en", "base", False
            )

        assert mock_batched.transcribe.call_count == 1, (
            f"batched.transcribe llamado {mock_batched.transcribe.call_count} veces, esperado 1"
        )


class TestTranscriptionResultMapsByTimestampOverlap:
    """El texto de cada segmento corresponde a los chunks whisper que overlappean."""

    def test_text_mapped_by_overlap(self):
        """Segmento [0, 3.5] debe obtener texto del whisper seg [0, 3.5]."""
        from speechlib.transcribe import transcribe_full_aligned

        diarization_segments = [
            [0.0, 3.5, "SPEAKER_00"],
            [3.5, 7.0, "SPEAKER_01"],
            [7.0, 10.0, "SPEAKER_00"],
        ]
        whisper_segs = _make_whisper_segments()
        mock_model = _mock_whisper_model()
        mock_batched = _mock_batched_pipeline(whisper_segs)

        with (
            patch("speechlib.transcribe.WhisperModel", return_value=mock_model),
            patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched),
        ):
            result = transcribe_full_aligned(
                "audio.wav", diarization_segments, "en", "base", False
            )

        assert result[0][2] == "Hello world"
        assert result[1][2] == "how are you"
        assert result[2][2] == "goodbye"

    def test_partial_overlap_concatenates_text(self):
        """Segmento [2.0, 5.0] debe obtener texto de whisper segs que overlappean parcialmente."""
        from speechlib.transcribe import transcribe_full_aligned

        diarization_segments = [
            [2.0, 5.0, "SPEAKER_00"],
        ]
        whisper_segs = _make_whisper_segments()  # [0-3.5], [3.5-7], [7-10]
        mock_model = _mock_whisper_model()
        mock_batched = _mock_batched_pipeline(whisper_segs)

        with (
            patch("speechlib.transcribe.WhisperModel", return_value=mock_model),
            patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched),
        ):
            result = transcribe_full_aligned(
                "audio.wav", diarization_segments, "en", "base", False
            )

        # Overlaps with seg1 [0-3.5] and seg2 [3.5-7]
        assert "Hello world" in result[0][2]
        assert "how are you" in result[0][2]


class TestFullFileTranscriptionReturnsSameFormat:
    """El resultado tiene formato [[start, end, text, speaker], ...]."""

    def test_format_is_list_of_4_element_lists(self):
        from speechlib.transcribe import transcribe_full_aligned

        diarization_segments = [
            [0.0, 3.5, "SPEAKER_00"],
            [7.0, 10.0, "SPEAKER_01"],
        ]
        whisper_segs = _make_whisper_segments()
        mock_model = _mock_whisper_model()
        mock_batched = _mock_batched_pipeline(whisper_segs)

        with (
            patch("speechlib.transcribe.WhisperModel", return_value=mock_model),
            patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched),
        ):
            result = transcribe_full_aligned(
                "audio.wav", diarization_segments, "en", "base", False
            )

        assert isinstance(result, list)
        for item in result:
            assert len(item) == 4, f"Esperado 4 elementos, obtenido {len(item)}: {item}"
            start, end, text, speaker = item
            assert isinstance(start, float)
            assert isinstance(end, float)
            assert isinstance(text, str)
            assert isinstance(speaker, str)

    def test_speaker_labels_preserved(self):
        from speechlib.transcribe import transcribe_full_aligned

        diarization_segments = [
            [0.0, 3.5, "SPEAKER_00"],
            [7.0, 10.0, "SPEAKER_01"],
        ]
        whisper_segs = _make_whisper_segments()
        mock_model = _mock_whisper_model()
        mock_batched = _mock_batched_pipeline(whisper_segs)

        with (
            patch("speechlib.transcribe.WhisperModel", return_value=mock_model),
            patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_batched),
        ):
            result = transcribe_full_aligned(
                "audio.wav", diarization_segments, "en", "base", False
            )

        assert result[0][3] == "SPEAKER_00"
        assert result[1][3] == "SPEAKER_01"
