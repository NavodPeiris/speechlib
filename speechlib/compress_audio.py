import logging
from pathlib import Path

from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from .step_timer import timed

logger = logging.getLogger(__name__)


@timed("compress_audio")
def compress_audio(source: Path, output: Path) -> Path | None:
    """Produce archival AAC copy: mono 96kbps 16kHz.

    Uses torchcodec (PyTorch-native) instead of FFmpeg subprocess.
    Audio is already loudnorm-normalized from earlier pipeline step.

    Returns output path on success, None on failure.
    """
    try:
        decoder = AudioDecoder(str(source))
        result = decoder.get_all_samples()
        encoder = AudioEncoder(result.data, sample_rate=result.sample_rate)
        encoder.to_file(str(output), bit_rate=96_000, num_channels=1, sample_rate=16_000)
        return output
    except Exception:
        logger.exception("compress_audio failed for %s", source)
        return None
