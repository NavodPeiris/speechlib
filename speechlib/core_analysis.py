import os
from pyannote.audio import Pipeline
import time
from .wav_segmenter import wav_file_segmentation
from .transcribe import transcribe_full_aligned
import torch
from functools import lru_cache
from .step_timer import measure, print_report
from .kernel_profiler import measure as kmeasure, print_report as kprint_report

import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["sox"]

from .speaker_recognition import speaker_recognition
from .write_log_file import write_log_file
from .segment_merger import merge_short_turns, merge_transcript_turns, group_by_sentences, group_by_speaker

from pathlib import Path
from .audio_state import AudioState
from .re_encode import re_encode
from .convert_to_mono import convert_to_mono
from .convert_to_wav import convert_to_wav
from .resample_to_16k import resample_to_16k
from .loudnorm import loudnorm
from .enhance_audio import enhance_audio


@lru_cache(maxsize=1)
def _get_diarization_pipeline(token: str):
    """Cache el pipeline de diarizacion para evitar recargarlo en cada llamada."""
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    return pipeline


# by default use google speech-to-text API
# if False, then use whisper finetuned version for sinhala
def core_analysis(
    file_name,
    voices_folder,
    log_folder,
    language,
    modelSize,
    ACCESS_TOKEN,
    model_type,
    quantization=False,
    custom_model_path=None,
    hf_model_id=None,
    aai_api_key=None,
    output_format: str = "txt",
    skip_enhance: bool = False,
    grouping_mode: str = "sentences",
):
    if log_folder is None:
        log_folder = os.path.join(
            os.path.dirname(os.path.abspath(file_name)), "output"
        )

    # <-------------------PreProcessing file-------------------------->

    state = AudioState(source_path=Path(file_name), working_path=Path(file_name))
    state = convert_to_wav(state)
    state = convert_to_mono(state)
    state = re_encode(state)
    state = resample_to_16k(state)
    state = loudnorm(state)
    if not skip_enhance:
        state = enhance_audio(state)

    # <--------------------running analysis--------------------------->

    speaker_tags = []

    pipeline = _get_diarization_pipeline(ACCESS_TOKEN)

    waveform, sample_rate = torchaudio.load(str(state.working_path))
    print("running diarization...")
    with measure("diarization", gpu=True), kmeasure("diarization"):
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    print("diarization done.")

    speakers = {}

    common = []

    # create a dictionary of SPEAKER_XX to real name mappings
    speaker_map = {}

    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )
    for turn, _, speaker in annotation.itertracks(yield_label=True):
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
            spk_name = speaker_recognition(
                str(state.working_path), voices_folder, spk_segments, identified
            )
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
            if (
                spk_tag1 not in merged
                and spk_tag2 not in merged
                and spk_tag1 != spk_tag2
                and speaker_map[spk_tag1] == speaker_map[spk_tag2]
            ):
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

    # merge short consecutive turns from the same speaker
    common = merge_short_turns(common)
    speakers = {}
    for segment in common:
        spk = segment[2]
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(segment)

    # transcribing the texts differently according to speaker
    print("running transcription...")
    with measure("transcription", gpu=True):
        if model_type == "faster-whisper":
            common_segments = transcribe_full_aligned(
                str(state.working_path), common, language, modelSize, quantization
            )
        else:
            for spk_tag, spk_segments in speakers.items():
                spk = speaker_map.get(spk_tag, spk_tag)
                segment_out = wav_file_segmentation(
                    str(state.working_path),
                    spk_segments,
                    language,
                    modelSize,
                    model_type,
                    quantization,
                    custom_model_path,
                    hf_model_id,
                    aai_api_key,
                )
                speakers[spk_tag] = segment_out
    print("transcription done.")

    if model_type != "faster-whisper":
        common_segments = []
        for item in common:
            speaker = item[2]
            start = item[0]
            end = item[1]

            for spk_tag, spk_segments in speakers.items():
                if speaker == speaker_map.get(spk_tag, spk_tag):
                    for segment in spk_segments:
                        if start == segment[0] and end == segment[1]:
                            common_segments.append([start, end, segment[2], speaker])

    # group post-transcription segments according to grouping_mode
    if model_type == "faster-whisper":
        if grouping_mode == "sentences":
            common_segments = group_by_sentences(common_segments)
        else:
            common_segments = group_by_speaker(common_segments)

    # writing log file
    with measure("write_log_file"):
        write_log_file(common_segments, log_folder, str(state.working_path), language, output_format)

    print_report()
    kprint_report()
    return common_segments
