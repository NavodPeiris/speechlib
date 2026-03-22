"""Forced alignment con Wav2Vec2 CTC — port simplificado de WhisperX."""
from dataclasses import dataclass
from functools import lru_cache

import torch
import torchaudio

SAMPLE_RATE = 16_000

_TORCHAUDIO_MODELS = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
}


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float


@dataclass
class AlignedSegment:
    start: float
    end: float
    text: str
    words: list


# ── Model loading ───────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _load_align_model(language: str, device: str):
    """Carga modelo Wav2Vec2 CTC desde torchaudio pipelines."""
    pipeline_name = _TORCHAUDIO_MODELS.get(language)
    if pipeline_name is None:
        raise ValueError(f"No alignment model for language: {language}")

    bundle = getattr(torchaudio.pipelines, pipeline_name)
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    dictionary = {c.lower(): i for i, c in enumerate(labels)}

    blank_id = 0
    for char, code in dictionary.items():
        if char == "[pad]" or char == "<pad>":
            blank_id = code

    return model, dictionary, blank_id


# ── Core CTC alignment (ported from WhisperX) ──────────────────────────────

def _get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def _backtrack(trellis, emission, tokens, blank_id=0):
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        path.append((j - 1, t - 1, prob))

        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None

    return path[::-1]


def _merge_repeats(path, transcript):
    segments = []
    i1, i2 = 0, 0
    while i1 < len(path):
        while i2 < len(path) and path[i1][0] == path[i2][0]:
            i2 += 1
        score = sum(path[k][2] for k in range(i1, i2)) / (i2 - i1)
        segments.append({
            "char": transcript[path[i1][0]],
            "start": path[i1][1],
            "end": path[i2 - 1][1] + 1,
            "score": score,
        })
        i1 = i2
    return segments


# ── Main entry point ────────────────────────────────────────────────────────

def align_words(audio_path, whisper_segments, language, device):
    """Refina word timestamps de Whisper usando forced alignment Wav2Vec2 CTC.

    Si el idioma no está soportado o alignment falla, retorna segmentos originales.
    """
    if language not in _TORCHAUDIO_MODELS:
        return whisper_segments

    try:
        model, dictionary, blank_id = _load_align_model(language, device)
    except Exception:
        return whisper_segments

    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    result = []
    for seg in whisper_segments:
        words = getattr(seg, "words", None) or []
        if not words:
            result.append(seg)
            continue

        aligned_words = _align_segment_words(
            waveform, seg, words, model, dictionary, blank_id, device,
        )

        if aligned_words is not None and len(aligned_words) == len(words):
            result.append(AlignedSegment(
                start=seg.start, end=seg.end, text=seg.text, words=aligned_words,
            ))
        else:
            # Fallback: keep original whisper timestamps
            result.append(seg)

    return result


def _align_segment_words(waveform, seg, words, model, dictionary, blank_id, device):
    """Alinea las palabras de un segmento Whisper usando CTC forced alignment."""
    f1 = int(seg.start * SAMPLE_RATE)
    f2 = int(seg.end * SAMPLE_RATE)
    segment_audio = waveform[:, f1:f2]

    if segment_audio.shape[-1] < 400:
        segment_audio = torch.nn.functional.pad(
            segment_audio, (0, 400 - segment_audio.shape[-1])
        )

    # Build cleaned text: lowercase, spaces→|, filter to dictionary chars
    raw_text = seg.text.strip()
    clean_chars = []
    clean_indices = []  # index into raw_text
    for i, ch in enumerate(raw_text):
        c = ch.lower().replace(" ", "|")
        if c in dictionary:
            clean_chars.append(c)
            clean_indices.append(i)

    if not clean_chars:
        return None

    tokens = [dictionary[c] for c in clean_chars]

    # Forward pass
    with torch.inference_mode():
        emissions, _ = model(segment_audio.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu()

    # Trellis + backtrack
    trellis = _get_trellis(emission, tokens, blank_id)
    path = _backtrack(trellis, emission, tokens, blank_id)

    if path is None:
        return None

    char_segments = _merge_repeats(path, "".join(clean_chars))

    # Convert frame indices to absolute timestamps
    duration = seg.end - seg.start
    num_frames = trellis.size(0) - 1
    ratio = duration / num_frames if num_frames > 0 else 0

    # Group characters into words (split on |)
    aligned_words = []
    current_word_chars = []
    current_word_text_parts = []

    for cseg in char_segments:
        if cseg["char"] == "|":
            if current_word_chars:
                w_start = seg.start + current_word_chars[0]["start"] * ratio
                w_end = seg.start + current_word_chars[-1]["end"] * ratio
                aligned_words.append(AlignedWord(
                    word="".join(current_word_text_parts),
                    start=round(w_start, 3),
                    end=round(w_end, 3),
                ))
                current_word_chars = []
                current_word_text_parts = []
        else:
            current_word_chars.append(cseg)
            current_word_text_parts.append(cseg["char"])

    # Flush last word
    if current_word_chars:
        w_start = seg.start + current_word_chars[0]["start"] * ratio
        w_end = seg.start + current_word_chars[-1]["end"] * ratio
        aligned_words.append(AlignedWord(
            word="".join(current_word_text_parts),
            start=round(w_start, 3),
            end=round(w_end, 3),
        ))

    # Preserve original word text (capitalization, punctuation) from Whisper
    if len(aligned_words) == len(words):
        for aw, orig in zip(aligned_words, words):
            aw.word = orig.word.strip()

    return aligned_words
