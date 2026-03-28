import logging
import os
from pathlib import Path
from typing import Union
import numpy as np
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cosine
import torch
from .audio_utils import slice_and_save
from .diarization import get_diarization_pipeline

logger = logging.getLogger(__name__)

SPEAKER_SIMILARITY_THRESHOLD = 0.40
VOICES_SKIP_PREFIX = "_"

_embedding_model = None
_inference = None


def _get_inference():
    global _embedding_model, _inference
    if _embedding_model is None:
        _embedding_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=os.environ.get("HF_TOKEN", None)
        )
        if torch.cuda.is_available():
            _embedding_model.to(torch.device("cuda"))
        _inference = Inference(_embedding_model, window="whole")
    return _inference


def get_embedding(audio_path: str) -> np.ndarray:
    inference = _get_inference()
    embedding = inference(audio_path)
    return embedding


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = np.asarray(emb1).flatten()
    emb2 = np.asarray(emb2).flatten()
    return 1.0 - cosine(emb1, emb2)


def find_best_speaker(
    test_embedding: np.ndarray,
    speaker_embeddings: dict[str, np.ndarray],
    threshold: float = SPEAKER_SIMILARITY_THRESHOLD,
) -> str:
    best_speaker = "unknown"
    best_score = -1.0

    scores = {}
    for speaker, emb in speaker_embeddings.items():
        score = cosine_similarity(test_embedding, emb)
        scores[speaker] = score
        if score > best_score:
            best_score = score
            best_speaker = speaker

    logger.debug(
        "Speaker scores: %s  best=%.3f threshold=%.2f", scores, best_score, threshold
    )

    if best_score < threshold:
        return "unknown"
    return best_speaker


def load_voice_embeddings(voices_folder: Path) -> dict[str, list[np.ndarray]]:
    """Carga embeddings por archivo para cada speaker en voices_folder.

    Retorna {speaker_name: [embedding_por_archivo, ...]}
    Omite directorios con prefijo VOICES_SKIP_PREFIX ('_').
    """
    result: dict[str, list[np.ndarray]] = {}
    voices_folder = Path(voices_folder)
    for entry in sorted(voices_folder.iterdir()):
        if not entry.is_dir() or entry.name.startswith(VOICES_SKIP_PREFIX):
            continue
        embs = []
        for wav in sorted(entry.glob("*.wav")):
            try:
                embs.append(get_embedding(str(wav)))
            except Exception as e:
                print(f"Error extracting embedding from {wav}: {e}")
        if embs:
            result[entry.name] = embs
    return result


def load_avg_voice_embeddings(voices_folder: Path) -> dict[str, np.ndarray]:
    """Carga embedding promedio por speaker en voices_folder.

    Retorna {speaker_name: avg_embedding}.
    """
    raw = load_voice_embeddings(voices_folder)
    return {name: np.mean(embs, axis=0) for name, embs in raw.items()}


def speaker_recognition(
    file_name,
    voices_folder,
    segments,
    wildcards,
    threshold: float = SPEAKER_SIMILARITY_THRESHOLD,
):
    inference = _get_inference()

    speaker_embeddings = load_avg_voice_embeddings(Path(voices_folder))

    folder_name = str(Path(file_name).parent / "tmp")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    limit = 60_000  # 60 segundos expresado en ms
    duration = 0
    collected_embeddings = []

    for i, segment in enumerate(segments, 1):
        start_ms = segment[0] * 1000
        end_ms = segment[1] * 1000
        file = (
            folder_name
            + "/"
            + os.path.splitext(os.path.basename(file_name))[0]
            + "_segment"
            + str(i)
            + ".wav"
        )
        slice_and_save(file_name, start_ms, end_ms, file)

        try:
            emb = inference(file)
            collected_embeddings.append(np.asarray(emb).flatten())
        except Exception as e:
            print(f"Error extracting embedding from segment: {e}")
            try:
                os.remove(file)
            except OSError:
                pass
            continue

        os.remove(file)

        duration += end_ms - start_ms
        if duration >= limit:
            break

    if not collected_embeddings:
        return "unknown"

    avg_emb = np.mean(collected_embeddings, axis=0)
    return find_best_speaker(avg_emb, speaker_embeddings, threshold)


def detect_unknown_speakers(
    audio_path: Union[str, Path],
    voices_folder: Union[str, Path],
    hf_token: str | None = None,
    threshold: float = SPEAKER_SIMILARITY_THRESHOLD,
    limit_s: float = 60.0,
) -> dict[str, list[list[float]]]:
    """Diariza el audio y retorna segmentos de speakers no reconocidos en voices_folder.

    Args:
        audio_path: WAV procesado (enhanced/16k).
        voices_folder: Carpeta con subdirectorios por speaker conocido.
        hf_token: Token HuggingFace para cargar el pipeline de diarización.
        threshold: Umbral de cosine similarity para considerar a un speaker conocido.
        limit_s: Segundos máximos de audio a analizar por speaker (rendimiento).

    Returns:
        {speaker_tag: [[start_s, end_s], ...]} — solo speakers no reconocidos.
        Los tags son los SPEAKER_XX asignados por pyannote.
    """
    pipeline = get_diarization_pipeline(hf_token)

    import torchaudio

    waveform, sample_rate = torchaudio.load(str(audio_path))
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )

    # Construir dict de segmentos por speaker_tag
    speakers: dict[str, list[list[float]]] = {}
    for turn, _, spk_tag in annotation.itertracks(yield_label=True):
        start = round(turn.start, 1)
        end = round(turn.end, 1)
        if spk_tag not in speakers:
            speakers[spk_tag] = []
        speakers[spk_tag].append([start, end, spk_tag])

    # Identificar cada speaker contra la voices library
    result: dict[str, list[list[float]]] = {}
    for spk_tag, segments in speakers.items():
        name = speaker_recognition(
            str(audio_path), str(voices_folder), segments, [], threshold
        )
        if name == "unknown":
            result[spk_tag] = [[s[0], s[1]] for s in segments]

    return result
