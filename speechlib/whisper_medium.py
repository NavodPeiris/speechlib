import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

def whisper_medium(file, language):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

    model.to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    # Load the WAV file
    waveform, sample_rate = torchaudio.load(file)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)

    if sample_rate != 16000:
        wav4trans = resampler(waveform)
    else:
        wav4trans = waveform

    input_features = processor(wav4trans.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

