from pyannote.audio import Pipeline
from .hf_access import (ACCESS_TOKEN)
from .wav_segmenter import (wav_file_segmentation)

from .speaker_recognition import (speaker_recognition)
from .write_log_file import (write_log_file)


# by default use google speech-to-text API
# if False, then use whisper finetuned version for sinhala
def core_analysis(file_name, voices_folder, log_folder, language):

    # <-------------------Processing file-------------------------->

    speaker_tags = []

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=ACCESS_TOKEN)

    diarization = pipeline(file_name, min_speakers=0, max_speakers=10)

    speakers = {}

    common = []

    # create a dictionary of SPEAKER_XX to real name mappings
    speaker_map = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):

        start = round(turn.start, 1)
        end = round(turn.end, 1)
        common.append([start, end, speaker])

        # find different speakers
        if speaker not in speaker_tags:
            speaker_tags.append(speaker)
            speaker_map[speaker] = speaker
            speakers[speaker] = []

        speakers[speaker].append([start, end, speaker])

    if voices_folder != None:
        identified = []

        for spk_tag, spk_segments in speakers.items():
            spk_name = speaker_recognition(file_name, voices_folder, spk_segments, identified)
            spk = spk_name
            identified.append(spk)
            speaker_map[spk_tag] = spk
            
    keys_to_remove = []
    merged = []

    # merging same speakers
    for spk_tag1, spk_segments1 in speakers.items():
        for spk_tag2, spk_segments2 in speakers.items():
            if spk_tag1 not in merged and spk_tag2 not in merged and spk_tag1 != spk_tag2 and speaker_map[spk_tag1] == speaker_map[spk_tag2]:
                for segment in spk_segments2:
                    speakers[spk_tag1].append(segment)

                merged.append(spk_tag1)
                merged.append(spk_tag2)
                keys_to_remove.append(spk_tag2)
    
    # fixing the speaker names in common
    for segment in common:
        speaker = segment[2]
        segment[2] = speaker_map[speaker]

    for key in keys_to_remove:
        del speakers[key]
        del speaker_map[key]

    # transcribing the texts differently according to speaker
    for spk_tag, spk_segments in speakers.items():
        spk = speaker_map[spk_tag]
        segment_out = wav_file_segmentation(file_name, spk_segments, language)
        speakers[spk_tag] = segment_out

    common_segments = []

    for item in common:
        speaker = item[2]
        start = item[0]
        end = item[1]

        for spk_tag, spk_segments in speakers.items():
            if speaker == speaker_map[spk_tag]:
                for segment in spk_segments:
                    if start == segment[0] and end == segment[1]:
                        common_segments.append([start, end, segment[2], speaker])

    # writing log file
    write_log_file(common_segments, log_folder)  

    return common_segments
