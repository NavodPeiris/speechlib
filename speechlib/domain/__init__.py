"""Domain model: value objects puros, sin I/O ni dependencias externas."""

from .transcript import SpeakerIdentity, TranscriptSegment, Transcript

__all__ = ["SpeakerIdentity", "TranscriptSegment", "Transcript"]
