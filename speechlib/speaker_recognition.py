from speechbrain.inference import SpeakerRecognition
import os
from collections import defaultdict
import torch
from .audio_utils import slice_and_save

if torch.cuda.is_available():
    verification = SpeakerRecognition.from_hparams(
        run_opts={"device": "cuda"},
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )
else:
    verification = SpeakerRecognition.from_hparams(
        run_opts={"device": "cpu"},
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )


def speaker_recognition(file_name, voices_folder, segments, wildcards):
    speakers = os.listdir(voices_folder)

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
            + file_name.split("/")[-1].split(".")[0]
            + "_segment"
            + str(i)
            + ".wav"
        )
        slice_and_save(file_name, start_ms, end_ms, file)

        max_score = 0
        person = "unknown"

        for speaker in speakers:
            voices = os.listdir(voices_folder + "/" + speaker)

            for voice in voices:
                voice_file = voices_folder + "/" + speaker + "/" + voice

                try:
                    score, prediction = verification.verify_files(voice_file, file)
                    prediction = prediction[0].item()
                    score = score[0].item()

                    if prediction == True:
                        if score >= max_score:
                            max_score = score
                            speakerId = speaker.split(".")[0]
                            if speakerId not in wildcards:
                                person = speakerId
                except Exception as err:
                    print("error occured while speaker recognition: ", err)

        Id_count[person] += 1

        os.remove(file)

        current_pred = max(Id_count, key=Id_count.get)

        duration += end_ms - start_ms
        if duration >= limit and current_pred != "unknown":
            break

    most_common_Id = max(Id_count, key=Id_count.get)
    return most_common_Id
