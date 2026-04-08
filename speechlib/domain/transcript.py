"""
Domain model del transcript con identidad del speaker.

Aggregate raiz: Transcript -> tuple[TranscriptSegment] -> SpeakerIdentity.

Invariante critico anti-bug: SpeakerIdentity.label SIEMPRE devuelve el
diarization_tag cuando no hay recognized_name. Es imposible por construccion
que aparezca el literal "unknown" como etiqueta visible — el bug del
relabel_vtt --all-speakers no puede repetirse en este modelo.

Estilo GOOS-sin-mocks: value objects inmutables, sin I/O, sin dependencias
externas. Toda la logica es funcional pura y testeable sin mocks.
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpeakerIdentity:
    """Identidad de un speaker en un segmento.

    Combina la informacion de pyannote (diarization_tag, siempre presente)
    con el resultado del reconocimiento contra la libreria de voces
    (recognized_name + similarity, opcionales).
    """

    diarization_tag: str
    recognized_name: Optional[str] = None
    similarity: Optional[float] = None

    @property
    def is_identified(self) -> bool:
        return self.recognized_name is not None

    @property
    def label(self) -> str:
        """Etiqueta visible. Fallback al diarization_tag cuando no hay match.

        Invariante: nunca retorna el string literal 'unknown'.
        """
        return self.recognized_name or self.diarization_tag

    def with_recognition(
        self, name: Optional[str], similarity: Optional[float]
    ) -> "SpeakerIdentity":
        """Devuelve una copia con el resultado del reconocimiento aplicado.

        Pasar name=None equivale a marcar como no identificado conservando
        el diarization_tag (caso del fallo de threshold).
        """
        return replace(self, recognized_name=name, similarity=similarity)


@dataclass(frozen=True)
class TranscriptSegment:
    """Un fragmento de transcripcion con timestamps y speaker."""

    start_ms: int
    end_ms: int
    text: str
    speaker: SpeakerIdentity

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def with_speaker(self, speaker: SpeakerIdentity) -> "TranscriptSegment":
        return replace(self, speaker=speaker)


@dataclass(frozen=True)
class Transcript:
    """Aggregate raiz del transcript completo de un audio.

    segments es tuple (no list) para mantener inmutabilidad y hashability.
    """

    segments: tuple[TranscriptSegment, ...]
    audio_path: str
    language: str

    @property
    def diarization_tags(self) -> frozenset[str]:
        """Conjunto unico de SPEAKER_XX presentes en el transcript."""
        return frozenset(s.speaker.diarization_tag for s in self.segments)

    def with_segments(
        self, segments: tuple[TranscriptSegment, ...]
    ) -> "Transcript":
        return replace(self, segments=tuple(segments))

    # ── Persistencia ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "audio_path": self.audio_path,
            "language": self.language,
            "segments": [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "text": seg.text,
                    "speaker": {
                        "diarization_tag": seg.speaker.diarization_tag,
                        "recognized_name": seg.speaker.recognized_name,
                        "similarity": seg.speaker.similarity,
                    },
                }
                for seg in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported transcript schema version: {version} "
                f"(expected {SCHEMA_VERSION})"
            )
        segments = tuple(
            TranscriptSegment(
                start_ms=seg["start_ms"],
                end_ms=seg["end_ms"],
                text=seg["text"],
                speaker=SpeakerIdentity(
                    diarization_tag=seg["speaker"]["diarization_tag"],
                    recognized_name=seg["speaker"]["recognized_name"],
                    similarity=seg["speaker"]["similarity"],
                ),
            )
            for seg in data["segments"]
        )
        return cls(
            segments=segments,
            audio_path=data["audio_path"],
            language=data["language"],
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "Transcript":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
