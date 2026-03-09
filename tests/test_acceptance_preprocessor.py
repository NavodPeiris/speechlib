"""
AT: Verify PreProcessor public API preserves source file.
"""

from pathlib import Path
from speechlib.speechlib import PreProcessor
from conftest import make_wav


def test_preprocessor_full_chain_preserves_source(tmp_path):
    """
    PreProcessor methods chain correctly and never modify the source.
    """
    source = make_wav(tmp_path / "source.wav", channels=2, sampwidth=1)
    original = source.read_bytes()

    prep = PreProcessor()
    wav_path = prep.convert_to_wav(str(source))
    mono_path = prep.convert_to_mono(wav_path)
    enc_path = prep.re_encode(mono_path)

    assert source.read_bytes() == original
    assert Path(enc_path).exists()
    assert Path(enc_path) != source
