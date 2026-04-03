"""AT: el pipeline de preprocessing produce audio a 16kHz (Slice A)."""
import torchaudio
from conftest import make_wav
from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav
from speechlib.convert_to_mono import convert_to_mono
from speechlib.re_encode import re_encode
from speechlib.resample_to_16k import resample_to_16k


def test_preprocessing_pipeline_outputs_16khz(tmp_path):
    """AT: dado audio a 44100 Hz, el pipeline produce un archivo a 16000 Hz."""
    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)

    state = AudioState(source_path=wav, working_path=wav)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)

    assert state.is_16khz is True
    _, sr = torchaudio.load(str(state.working_path))
    assert sr == 16000


def test_preprocessing_pipeline_already_16khz(tmp_path):
    """AT: audio ya a 16kHz pasa el pipeline sin crear archivos _16k."""
    wav = make_wav(tmp_path / "audio.wav", framerate=16000, n_frames=1600)

    state = AudioState(source_path=wav, working_path=wav)
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)

    assert state.is_16khz is True
    _, sr = torchaudio.load(str(state.working_path))
    assert sr == 16000
    assert not any("_16k" in f.name for f in tmp_path.iterdir())
