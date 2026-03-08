import os
from .audio_utils import slice_and_save
from .transcribe import transcribe


def wav_file_segmentation(
    file_name,
    segments,
    language,
    modelSize,
    model_type,
    quantization,
    custom_model_path,
    hf_model_path,
    aai_api_key,
):
    trans = ""
    texts = []

    folder_name = "segments"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    i = 0

    for segment in segments:
        start_ms = segment[0] * 1000
        end_ms = segment[1] * 1000
        i = i + 1
        file = folder_name + "/" + "segment" + str(i) + ".wav"
        slice_and_save(file_name, start_ms, end_ms, file)

        try:
            trans = transcribe(
                file,
                language,
                modelSize,
                model_type,
                quantization,
                custom_model_path,
                hf_model_path,
                aai_api_key,
            )
            texts.append([segment[0], segment[1], trans])
        except Exception as err:
            print("ERROR while transcribing: ", err)
        os.remove(file)

    return texts
