import os
import torch
from pydub import AudioSegment
from collections import defaultdict
from speechbrain.inference import SpeakerRecognition
from .base import BaseRecognizer

_verification: SpeakerRecognition | None = None


class SpeechBrainRecognizer(BaseRecognizer):
    """Speaker recognition via SpeechBrain ECAPA-VoxCeleb (or compatible model)."""

    def __init__(self, model_id: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.model_id = model_id

    def _get_model(self) -> SpeakerRecognition:
        global _verification
        if _verification is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _verification = SpeakerRecognition.from_hparams(
                run_opts={"device": device},
                source=self.model_id,
                savedir=f"pretrained_models/{self.model_id.split('/')[-1]}",
            )
        return _verification

    def recognize(self, file_name: str, voices_folder: str, segments: list, identified: list) -> str:
        verification = self._get_model()
        speakers = os.listdir(voices_folder)
        id_count: dict = defaultdict(int)
        audio = AudioSegment.from_file(file_name, format="wav")

        folder_name = "temp"
        os.makedirs(folder_name, exist_ok=True)

        limit_ms = 60 * 1000
        duration_ms = 0

        for i, segment in enumerate(segments, 1):
            start_ms = int(segment[0] * 1000)
            end_ms = int(segment[1] * 1000)
            if end_ms <= start_ms:
                continue
            clip = audio[start_ms:end_ms]
            if len(clip) == 0:
                continue
            base = os.path.splitext(os.path.basename(file_name))[0]
            temp_file = f"{folder_name}/{base}_segment{i}.wav"
            clip.export(temp_file, format="wav")

            max_score = 0.0
            person = "unknown"

            for speaker in speakers:
                for voice in os.listdir(f"{voices_folder}/{speaker}"):
                    voice_file = f"{voices_folder}/{speaker}/{voice}"
                    try:
                        score, prediction = verification.verify_files(voice_file, temp_file)
                        if prediction[0].item() and score[0].item() >= max_score:
                            max_score = score[0].item()
                            candidate = speaker.split(".")[0]
                            if candidate not in identified:
                                person = candidate
                    except Exception as e:
                        print(f"error in speaker recognition: {e}")

            id_count[person] += 1
            os.remove(temp_file)

            duration_ms += end_ms - start_ms
            current_best = max(id_count, key=id_count.get)
            if duration_ms >= limit_ms and current_best != "unknown":
                break

        return max(id_count, key=id_count.get)
