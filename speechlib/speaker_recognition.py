import logging
import os
from pathlib import Path
import numpy as np
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cosine
import torch
from .audio_utils import slice_and_save

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

    logger.debug("Speaker scores: %s  best=%.3f threshold=%.2f", scores, best_score, threshold)

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


def speaker_recognition(file_name, voices_folder, segments, wildcards):
    inference = _get_inference()

    speaker_embeddings = load_avg_voice_embeddings(Path(voices_folder))

    from collections import defaultdict

    Id_count = defaultdict(int)

    folder_name = str(Path(file_name).parent / "tmp")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    i = 0

    limit = 60_000  # 60 segundos expresado en ms
    duration = 0

    for segment in segments:
        start_ms = segment[0] * 1000
        end_ms = segment[1] * 1000
        i = i + 1
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
            test_emb = inference(file)
        except Exception as e:
            print(f"Error extracting embedding from segment: {e}")
            try:
                os.remove(file)
            except OSError:
                pass
            continue

        best_speaker = find_best_speaker(test_emb, speaker_embeddings)

        if best_speaker != "unknown":
            speakerId = best_speaker.split(".")[0]
            if speakerId not in wildcards:
                Id_count[speakerId] += 1

        os.remove(file)

        current_pred = max(Id_count, key=Id_count.get) if Id_count else "unknown"

        duration += end_ms - start_ms
        if duration >= limit and current_pred != "unknown":
            break

    most_common_Id = max(Id_count, key=Id_count.get) if Id_count else "unknown"
    return most_common_Id
