from __future__ import annotations
from abc import ABC, abstractmethod


class BaseDiarizer(ABC):
    """
    Abstract base class for speaker diarization.

    A diarizer segments an audio stream by speaker identity, returning a
    time-ordered list of speaker turns without requiring any prior knowledge
    of who the speakers are.

    Implement this class to plug in any diarization backend.
    All provider-specific parameters (e.g. min/max speaker count, clustering
    thresholds) must be accepted in ``__init__`` — they must *not* be passed
    at call time.
    """

    @abstractmethod
    def diarize(self, file) -> list[tuple[float, float, str]]:
        """
        Segment ``waveform`` into speaker turns.

        Parameters
        ----------
        file : path to audio file

        Returns
        -------
        list[tuple[float, float, str]]
            Time-ordered list of ``(start_sec, end_sec, speaker_tag)`` tuples.
            ``speaker_tag`` is an opaque string label (e.g. ``"SPEAKER_00"``)
            that is consistent within a single call but has no meaning across
            calls.
        """


class BaseRecognizer(ABC):
    """
    Abstract base class for speaker recognition / identification.

    A recognizer maps an opaque diarization speaker tag to a real name by
    comparing audio segments against reference voice samples stored on disk.

    Implement this class to plug in any speaker-verification backend.
    All provider-specific parameters must be accepted in ``__init__``.
    """

    @abstractmethod
    def recognize(
        self,
        file_name: str,
        voices_folder: str,
        segments: list[list[float]],
        identified: list[str],
    ) -> str:
        """
        Identify the most likely speaker for the given audio segments.

        Parameters
        ----------
        file_name : str
            Path to the preprocessed mono 16-bit PCM WAV file.
        voices_folder : str
            Root directory that contains one sub-folder per known speaker.
            Each sub-folder is named after the speaker and holds one or more
            ``.wav`` reference recordings.
        segments : list[list[float]]
            Segments attributed to this speaker by the diarizer, each as
            ``[start_sec, end_sec, speaker_tag]``.
        identified : list[str]
            Speaker names already assigned to other speaker tags in this file.
            The recognizer must not return a name that already appears here
            (each tag must map to a unique person).

        Returns
        -------
        str
            The matched speaker name (a sub-folder name from ``voices_folder``),
            or ``"unknown"`` if no reference voice scores above the threshold.
        """


class BaseASR(ABC):
    """
    Abstract base class for automatic speech recognition (ASR).

    An ASR instance transcribes a short audio clip to text.  The pipeline
    calls ``transcribe`` once per diarized segment.

    Implement this class to plug in any ASR backend.
    All provider-specific parameters (e.g. model size, beam size, temperature,
    quantization) must be accepted in ``__init__`` — they must *not* be passed
    at call time.
    """

    @abstractmethod
    def transcribe(self, audio: str | BinaryIO, language: str | None) -> str:
        """
        Transcribe a single audio segment to text.

        Parameters
        ----------
        audio : str | BinaryIO
            Either a file-system path to a mono 16-bit PCM WAV clip, or a
            seekable in-memory buffer (e.g. ``io.BytesIO``) containing the
            same WAV data.  Callers pass a ``BytesIO`` buffer to avoid all
            disk I/O; implementations must handle both forms.
        language : str | None
            BCP-47 language code (e.g. ``"en"``, ``"fr"``), or ``None`` to
            trigger automatic language detection.

        Returns
        -------
        str
            Raw transcription text for the segment.  Return an empty string
            ``""`` if the segment contains no speech.
        """
