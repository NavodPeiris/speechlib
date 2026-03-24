"""
AT: AudioState expone artifacts_dir como carpeta oculta junto al source.
"""
from pathlib import Path
from speechlib.audio_state import AudioState


def test_artifacts_dir_is_hidden_folder_next_to_source():
    state = AudioState(source_path=Path("/rec/Voz 260320.m4a"), working_path=Path("/rec/Voz 260320.m4a"))
    assert state.artifacts_dir == Path("/rec/.Voz 260320")


def test_artifacts_dir_uses_source_stem_not_working():
    state = AudioState(source_path=Path("/rec/audio.m4a"), working_path=Path("/rec/audio_16k.wav"))
    assert state.artifacts_dir == Path("/rec/.audio")


def test_artifacts_dir_works_with_wav_source():
    state = AudioState(source_path=Path("/rec/meeting.wav"), working_path=Path("/rec/meeting.wav"))
    assert state.artifacts_dir == Path("/rec/.meeting")


def test_artifacts_dir_preserved_after_model_copy():
    state = AudioState(source_path=Path("/rec/audio.wav"), working_path=Path("/rec/audio.wav"))
    updated = state.model_copy(update={"working_path": Path("/rec/.audio/16k.wav"), "is_16khz": True})
    assert updated.artifacts_dir == state.artifacts_dir
