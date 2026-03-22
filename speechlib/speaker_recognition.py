import os
import numpy as np
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cosine
import torch
from .audio_utils import slice_and_save

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
    test_embedding: np.ndarray, speaker_embeddings: dict[str, np.ndarray]
) -> str:
    best_speaker = "unknown"
    best_score = -1.0

    for speaker, emb in speaker_embeddings.items():
        score = cosine_similarity(test_embedding, emb)
        if score > best_score:
            best_score = score
            best_speaker = speaker

    return best_speaker


def speaker_recognition(file_name, voices_folder, segments, wildcards):
    inference = _get_inference()

    speakers = os.listdir(voices_folder)

    speaker_embeddings = {}

    for speaker in speakers:
        speaker_path = os.path.join(voices_folder, speaker)
        if not os.path.isdir(speaker_path):
            continue

        voice_files = os.listdir(speaker_path)
        embeddings = []

        for voice_file in voice_files:
            voice_path = os.path.join(speaker_path, voice_file)
            try:
                emb = inference(voice_path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error extracting embedding from {voice_path}: {e}")

        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            speaker_embeddings[speaker] = avg_emb

    from collections import defaultdict

    Id_count = defaultdict(int)

    folder_name = "temp"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    i = 0

    limit = 60
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
            os.remove(file)
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
