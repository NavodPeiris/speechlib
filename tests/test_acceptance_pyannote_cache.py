"""AT: Pipeline.from_pretrained se cachea entre llamadas a core_analysis.

En lugar de cargar el pipeline de diarizacion en cada llamada a core_analysis,
se debe usar un cache (lru_cache) para reutilizar la instancia.
"""
import pytest
from unittest.mock import patch, MagicMock, call


@pytest.fixture(autouse=True)
def clear_diarization_cache():
    from speechlib.core_analysis import _get_diarization_pipeline
    _get_diarization_pipeline.cache_clear()
    yield
    _get_diarization_pipeline.cache_clear()


def _state(path):
    s = MagicMock()
    s.working_path = path
    return s


def _make_mock_pipeline():
    """Pipeline mock que devuelve un solo segmento."""
    turn = MagicMock()
    turn.start = 0.0
    turn.end = 1.0

    mock_annotation = MagicMock(spec=[])
    mock_annotation.itertracks = MagicMock(return_value=[
        (turn, None, "SPEAKER_00"),
    ])
    mock_annotation.write_rttm = MagicMock()

    pipeline_instance = MagicMock()
    pipeline_instance.return_value = mock_annotation
    pipeline_instance.to = MagicMock()
    return pipeline_instance


def _run_core_analysis(fake_wav, tmp_path, mock_from_pretrained):
    """Helper que ejecuta core_analysis con todos los mocks necesarios."""
    from speechlib.core_analysis import core_analysis

    pipeline_mock = _make_mock_pipeline()
    mock_from_pretrained.return_value = pipeline_mock

    def fake_full_aligned(file, segments, *args, **kwargs):
        return [[s[0], s[1], "text", s[2]] for s in segments]

    with (
        patch("speechlib.core_analysis.torchaudio") as mock_torchaudio,
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s),
        patch("speechlib.core_analysis.transcribe_full_aligned", side_effect=fake_full_aligned),
        patch("speechlib.core_analysis.wav_file_segmentation", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
    ):
        mock_torchaudio.load.return_value = (MagicMock(), 16000)
        return core_analysis(
            str(fake_wav), None, str(tmp_path), "en", "tiny", "TOKEN", "faster-whisper"
        )


def test_pipeline_loaded_once_for_multiple_core_analysis_calls(tmp_path):
    """Pipeline.from_pretrained se llama UNA vez aunque se procesen 2 archivos."""
    wav1 = tmp_path / "audio1.wav"
    wav1.write_bytes(b"RIFF")
    wav2 = tmp_path / "audio2.wav"
    wav2.write_bytes(b"RIFF")

    with patch("speechlib.diarization.Pipeline.from_pretrained") as mock_from_pretrained:
        _run_core_analysis(wav1, tmp_path, mock_from_pretrained)
        _run_core_analysis(wav2, tmp_path, mock_from_pretrained)

        assert mock_from_pretrained.call_count == 1, (
            f"Pipeline.from_pretrained llamado {mock_from_pretrained.call_count} veces, "
            "esperado 1 (debe estar cacheado)"
        )


def test_cached_pipeline_returns_same_instance(tmp_path):
    """El pipeline retornado por el cache es la misma instancia."""
    from speechlib.core_analysis import _get_diarization_pipeline

    mock_pipeline = _make_mock_pipeline()
    with patch("speechlib.diarization.Pipeline.from_pretrained", return_value=mock_pipeline):
        p1 = _get_diarization_pipeline("TOKEN")
        p2 = _get_diarization_pipeline("TOKEN")

    assert p1 is p2, "El cache debe retornar la misma instancia"
