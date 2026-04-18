import os
from pyannote.audio import Pipeline
import time
from .wav_segmenter import (wav_file_segmentation)
import torch, torchaudio

from .speaker_recognition import (speaker_recognition)
from .write_log_file import (write_log_file)

from .re_encode import (re_encode)
from .convert_to_mono import (convert_to_mono)
from .convert_to_wav import (convert_to_wav)

# Cache Pyannote pipeline to avoid reloading
_pipeline_cache = {}

# by default use google speech-to-text API
# if False, then use whisper finetuned version for sinhala
def core_analysis(file_name, voices_folder, log_folder, language, modelSize, ACCESS_TOKEN, model_type, quantization=False, custom_model_path=None, hf_model_id=None, aai_api_key=None, output_format="both", **kwargs):

    # <-------------------PreProcessing file-------------------------->

    print(f"Preprocessing {file_name}: Converting to WAV...")
    # check if file is in wav format, if not convert to wav
    file_name = convert_to_wav(file_name)

    print(f"Preprocessing {file_name}: Converting to Mono...")
    # convert file to mono
    convert_to_mono(file_name)

    print(f"Preprocessing {file_name}: Re-encoding to 16-bit PCM...")
    # re-encode file to 16-bit PCM encoding
    re_encode(file_name)

    # <--------------------running analysis--------------------------->

    speaker_tags = []
    
    global _pipeline_cache
    if ACCESS_TOKEN not in _pipeline_cache:
        print("Loading pyannote diarization pipeline...")
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=ACCESS_TOKEN)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        else:
            device = torch.device("cpu")

        pipe.to(device)
        _pipeline_cache[ACCESS_TOKEN] = pipe
        
    pipeline = _pipeline_cache[ACCESS_TOKEN]
    waveform, sample_rate = torchaudio.load(file_name)

    start_time = int(time.time())
    print("running diarization...")
    min_speakers = kwargs.get("min_speakers", 1)
    max_speakers = kwargs.get("max_speakers", 10)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, min_speakers=min_speakers, max_speakers=max_speakers)
    end_time = int(time.time())
    elapsed_time = int(end_time - start_time)
    print(f"diarization done. Time taken: {elapsed_time} seconds.")

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

    if voices_folder != None and voices_folder != "":
        identified = []

        start_time = int(time.time())
        print("running speaker recognition...")
        for spk_tag, spk_segments in speakers.items():
            spk_name = speaker_recognition(file_name, voices_folder, spk_segments, identified)
            spk = spk_name
            identified.append(spk)
            speaker_map[spk_tag] = spk
        end_time = int(time.time())
        elapsed_time = int(end_time - start_time)
        print(f"speaker recognition done. Time taken: {elapsed_time} seconds.")

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
    start_time = int(time.time())
    print("running transcription...")
    for spk_tag, spk_segments in speakers.items():
        spk = speaker_map[spk_tag]
        segment_out = wav_file_segmentation(file_name, spk_segments, language, modelSize, model_type, quantization, custom_model_path, hf_model_id, aai_api_key, **kwargs)
        speakers[spk_tag] = segment_out
    end_time = int(time.time())
    elapsed_time = int(end_time - start_time)
    print(f"transcription done. Time taken: {elapsed_time} seconds.")

    common_segments = []

    for item in common:
        speaker = item[2]
        start = item[0]
        end = item[1]

        for spk_tag, spk_segments in speakers.items():
            if speaker == speaker_map[spk_tag]:
                for segment in spk_segments:
                    if start == segment[0] and end == segment[1]:
                        common_segments.append({
                            "file_name": file_name,
                            "start_time": start,
                            "end_time": end,
                            "text": segment[2],
                            "speaker": speaker,
                            "model_used": modelSize if model_type in ["whisper", "faster-whisper"] else model_type,
                            "language_detected": language if language else "auto"
                        })

    # writing log file
    write_log_file(common_segments, log_folder, file_name, language, output_format)  

    return common_segments
