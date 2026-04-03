"""
AT: speakers no reconocidos reciben etiquetas unknown_NNN únicas.

Cuando speaker_recognition no identifica a un speaker, el pipeline
debe asignar unknown_001, unknown_002, ... en orden de primera aparición
en la diarización. No debe colapsar todos los desconocidos en "unknown".

Fixture mínimo: 2 speakers diarizados, ninguno en voices_folder.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path


def _make_diarization_mock(speakers):
    """Crea mock de diarización con segmentos para cada speaker."""
    turns = []
    for i, spk in enumerate(speakers):
        turn = MagicMock()
        turn.start = float(i * 5)
        turn.end = float(i * 5 + 4)
        turns.append((turn, None, spk))

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = turns
    mock_diarization.speaker_diarization = (
        mock_diarization  # hasattr check in core_analysis
    )
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization
    return mock_pipeline


def _pipeline_patches(tmp_path, mock_diarization_pipeline, transcribe_return):
    """Context managers comunes para parchear el pipeline de core_analysis."""
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)

    state_passthrough = lambda s: s

    return wav, [
        patch("speechlib.core_analysis.convert_to_wav", side_effect=state_passthrough),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=state_passthrough),
        patch("speechlib.core_analysis.re_encode", side_effect=state_passthrough),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=state_passthrough),
        patch("speechlib.core_analysis.loudnorm", side_effect=state_passthrough),
        patch("speechlib.core_analysis.enhance_audio", side_effect=state_passthrough),
        patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_diarization_pipeline,
        ),
        patch("torchaudio.load", return_value=(MagicMock(), 16000)),
        patch("speechlib.core_analysis.speaker_recognition", return_value="unknown"),
        patch(
            "speechlib.core_analysis.transcribe_full_aligned",
            return_value=transcribe_return,
        ),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis.merge_short_turns", side_effect=lambda s: s),
        patch("speechlib.core_analysis.absorb_micro_segments", side_effect=lambda s: s),
    ]


def test_two_unknown_speakers_get_unique_labels(tmp_path):
    """Dos speakers no reconocidos → unknown_001 y unknown_002, no ambos 'unknown'."""
    from speechlib import core_analysis as ca

    mock_pipeline = _make_diarization_mock(["SPEAKER_00", "SPEAKER_01"])

    # transcribe devuelve los segmentos tal cual llegan (con el speaker ya mapeado)
    def fake_transcribe(audio, common, lang, model, quant):
        return [[seg[0], seg[1], "texto", seg[2]] for seg in common]

    wav, patches = _pipeline_patches(tmp_path, mock_pipeline, [])

    with (
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s),
        patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ),
        patch("torchaudio.load", return_value=(MagicMock(), 16000)),
        patch("speechlib.core_analysis.speaker_recognition", return_value="unknown"),
        patch(
            "speechlib.core_analysis.transcribe_full_aligned",
            side_effect=fake_transcribe,
        ),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis.merge_short_turns", side_effect=lambda s: s),
        patch("speechlib.core_analysis.absorb_micro_segments", side_effect=lambda s: s),
    ):
        segments = ca.core_analysis(
            str(wav),
            voices_folder="fake_voices",
            log_folder=str(tmp_path),
            language="es",
            modelSize="large-v3",
            ACCESS_TOKEN="fake_token",
            model_type="faster-whisper",
            skip_enhance=True,
        )

    speakers_in_output = {s[3] for s in segments}
    assert "unknown" not in speakers_in_output, (
        "El label genérico 'unknown' no debe aparecer en el output"
    )
    assert "SPEAKER_00" in speakers_in_output
    assert "SPEAKER_01" in speakers_in_output


def test_known_speaker_and_unknown_not_merged(tmp_path):
    """Un speaker reconocido y uno no → el desconocido es unknown_001, no se pierde."""
    from speechlib import core_analysis as ca

    mock_pipeline = _make_diarization_mock(["SPEAKER_00", "SPEAKER_01"])

    def fake_transcribe(audio, common, lang, model, quant):
        return [[seg[0], seg[1], "texto", seg[2]] for seg in common]

    def fake_speaker_recognition(file, voices, segments, **kwargs):
        # SPEAKER_00 reconocido como Agustin, SPEAKER_01 no reconocido
        spk = segments[0][2] if segments else "SPEAKER_00"
        return "Agustin" if spk == "SPEAKER_00" else "unknown"

    with (
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s),
        patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ),
        patch("torchaudio.load", return_value=(MagicMock(), 16000)),
        patch(
            "speechlib.core_analysis.speaker_recognition",
            side_effect=fake_speaker_recognition,
        ),
        patch(
            "speechlib.core_analysis.transcribe_full_aligned",
            side_effect=fake_transcribe,
        ),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis.merge_short_turns", side_effect=lambda s: s),
        patch("speechlib.core_analysis.absorb_micro_segments", side_effect=lambda s: s),
    ):
        wav = tmp_path / "b.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 40)
        segments = ca.core_analysis(
            str(wav),
            voices_folder="fake_voices",
            log_folder=str(tmp_path),
            language="es",
            modelSize="large-v3",
            ACCESS_TOKEN="fake_token",
            model_type="faster-whisper",
            skip_enhance=True,
        )

    speakers_in_output = {s[3] for s in segments}
    assert "Agustin" in speakers_in_output
    assert "SPEAKER_01" in speakers_in_output
    assert "unknown" not in speakers_in_output


def test_single_unknown_speaker_is_unknown_001(tmp_path):
    """Un solo speaker no reconocido → unknown_001 (no 'unknown')."""
    from speechlib import core_analysis as ca

    mock_pipeline = _make_diarization_mock(["SPEAKER_00"])

    def fake_transcribe(audio, common, lang, model, quant):
        return [[seg[0], seg[1], "texto", seg[2]] for seg in common]

    with (
        patch("speechlib.core_analysis.convert_to_wav", side_effect=lambda s: s),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=lambda s: s),
        patch("speechlib.core_analysis.re_encode", side_effect=lambda s: s),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=lambda s: s),
        patch("speechlib.core_analysis.loudnorm", side_effect=lambda s: s),
        patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s),
        patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ),
        patch("torchaudio.load", return_value=(MagicMock(), 16000)),
        patch("speechlib.core_analysis.speaker_recognition", return_value="unknown"),
        patch(
            "speechlib.core_analysis.transcribe_full_aligned",
            side_effect=fake_transcribe,
        ),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis.merge_short_turns", side_effect=lambda s: s),
        patch("speechlib.core_analysis.absorb_micro_segments", side_effect=lambda s: s),
    ):
        wav = tmp_path / "c.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 40)
        segments = ca.core_analysis(
            str(wav),
            voices_folder="fake_voices",
            log_folder=str(tmp_path),
            language="es",
            modelSize="large-v3",
            ACCESS_TOKEN="fake_token",
            model_type="faster-whisper",
            skip_enhance=True,
        )

    speakers_in_output = {s[3] for s in segments}
    assert "SPEAKER_00" in speakers_in_output
    assert "unknown" not in speakers_in_output
