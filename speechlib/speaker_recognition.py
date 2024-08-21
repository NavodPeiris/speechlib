from speechbrain.pretrained import SpeakerRecognition
import os
from pydub import AudioSegment
from collections import defaultdict
import torch

if torch.cuda.is_available():
    verification = SpeakerRecognition.from_hparams(run_opts={"device":"cuda"}, source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
else:
    verification = SpeakerRecognition.from_hparams(run_opts={"device":"cpu"}, source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

# recognize speaker name
def speaker_recognition(file_name, voices_folder, segments, wildcards):

    speakers = os.listdir(voices_folder)

    Id_count = defaultdict(int)
    # Load the WAV file
    audio = AudioSegment.from_file(file_name, format="wav")

    folder_name = "temp"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    i = 0
    
    '''
    iterate over segments and check speaker for increased accuracy.
    assign speaker name to arbitrary speaker tag 'SPEAKER_XX'
    '''

    limit = 60
    duration = 0

    for segment in segments:
        start = segment[0] * 1000   # start time in miliseconds
        end = segment[1] * 1000     # end time in miliseconds
        clip = audio[start:end]
        i = i + 1
        file = folder_name + "/" + file_name.split("/")[-1].split(".")[0] + "_segment"+ str(i) + ".wav"
        clip.export(file, format="wav")

        max_score = 0
        person = "unknown"      # if no match to any voice, then return unknown

        for speaker in speakers:

            voices = os.listdir(voices_folder + "/" + speaker)

            for voice in voices:
                voice_file = voices_folder + "/" + speaker + "/" + voice

                try:
                    # compare voice file with audio file
                    score, prediction = verification.verify_files(voice_file, file)
                    prediction = prediction[0].item()
                    score = score[0].item()

                    if prediction == True:
                        if score >= max_score:
                            max_score = score
                            speakerId = speaker.split(".")[0]  
                            if speakerId not in wildcards:        # speaker_00 cannot be speaker_01
                                person = speakerId
                except Exception as err:
                    print("error occured while speaker recognition: ", err)

        Id_count[person] += 1

        # Delete the WAV file after processing
        os.remove(file)

        current_pred = max(Id_count, key=Id_count.get)

        duration += (end - start)
        if duration >= limit and current_pred != "unknown":
            break
    
    most_common_Id = max(Id_count, key=Id_count.get)
    return most_common_Id

