"""Domain model: value objects puros, sin I/O ni dependencias externas."""

from .sample_extraction import (
    SampleClip,
    SpeakerSamplePlan,
    plan_speaker_samples,
)
from .transcript import SpeakerIdentity, TranscriptSegment, Transcript

__all__ = [
    "SampleClip",
    "SpeakerIdentity",
    "SpeakerSamplePlan",
    "Transcript",
    "TranscriptSegment",
    "plan_speaker_samples",
]
