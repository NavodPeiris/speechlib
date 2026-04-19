from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from pydub import AudioSegment

from .components.base import BaseDiarizer, BaseRecognizer, BaseASR
from .re_encode import re_encode
from .convert_to_mono import convert_to_mono
from .convert_to_wav import convert_to_wav
from .write_log_file import write_log_file


def _preprocess(file: str, verbose: bool) -> str:
    if verbose:
        print(f"Preprocessing {file}: converting to WAV...")
    file = convert_to_wav(file, verbose=verbose)
    if verbose:
        print(f"Preprocessing {file}: converting to mono...")
    convert_to_mono(file, verbose=verbose)
    if verbose:
        print(f"Preprocessing {file}: re-encoding to 16-bit PCM...")
    re_encode(file, verbose=verbose)
    return file


def _process_audio(
    file: str,
    diarizer: BaseDiarizer,
    recognizer: BaseRecognizer | None,
    asr_model: BaseASR,
    language: str | None,
    voices_folder: str | None,
    verbose: bool,
    workers: int,
) -> list[dict]:
    """
    Run diarize → (recognize) → transcribe on an already-preprocessed WAV.

    Diarization and speaker recognition always run sequentially on the full
    file — splitting audio before diarization destroys global speaker context.
    Transcription of individual segments is parallelised across ``workers``
    threads when ``workers > 1``; all heavy inference (CTranslate2, PyTorch)
    releases the GIL so threads yield real concurrency.
    """
    # --- Diarization (full file, single pass) ---
    t0 = time.time()
    print("Running diarization...")
    raw_segments = diarizer.diarize(file)
    if verbose:
        print(f"Diarization done in {int(time.time() - t0)}s.")

    speaker_tags: list[str] = []
    speakers: dict[str, list] = {}
    speaker_map: dict[str, str] = {}

    for start, end, tag in raw_segments:
        if tag not in speaker_tags:
            speaker_tags.append(tag)
            speaker_map[tag] = tag
            speakers[tag] = []
        speakers[tag].append([start, end, tag])

    # --- Optional speaker recognition (full file, single pass) ---
    if recognizer is not None and voices_folder:
        identified: list[str] = []
        t0 = time.time()
        print("Running speaker recognition...")
        for tag, segs in speakers.items():
            name = recognizer.recognize(file, voices_folder, segs, identified)
            identified.append(name)
            speaker_map[tag] = name
        if verbose:
            print(f"Speaker recognition done in {int(time.time() - t0)}s.")

    # Merge tags that resolved to the same real name
    keys_to_remove: list[str] = []
    merged_tags: list[str] = []
    for t1, segs1 in speakers.items():
        for t2, segs2 in speakers.items():
            if (
                t1 not in merged_tags and t2 not in merged_tags
                and t1 != t2
                and speaker_map[t1] == speaker_map[t2]
            ):
                segs1.extend(segs2)
                merged_tags += [t1, t2]
                keys_to_remove.append(t2)

    ordered = [(s, e, speaker_map[tag]) for s, e, tag in raw_segments]

    for key in keys_to_remove:
        del speakers[key]
        del speaker_map[key]

    # --- Transcription (parallel across segments) ---
    t0 = time.time()
    print("Running transcription...")
    audio = AudioSegment.from_file(file, format="wav")
    model_label = getattr(asr_model, "model_size", type(asr_model).__name__)

    # Extract all clips into BytesIO buffers up-front (fast, no disk I/O).
    work: list[tuple[BytesIO, str, float, float]] = []  # (buf, tag, start, end)
    for tag, segs in speakers.items():
        for seg in segs:
            start_ms = int(seg[0] * 1000)
            end_ms = int(seg[1] * 1000)
            if end_ms <= start_ms:
                continue
            clip = audio[start_ms:end_ms]
            if len(clip) == 0:
                continue
            buf = BytesIO()
            clip.export(buf, format="wav")
            buf.seek(0)
            work.append((buf, tag, seg[0], seg[1]))

    def _transcribe(item: tuple) -> tuple[str, float, float, str]:
        buf, tag, start, end = item
        try:
            text = asr_model.transcribe(buf, language)
        except Exception as e:
            print(f"ERROR transcribing segment [{start}-{end}]: {e}")
            text = ""
        return tag, start, end, text

    total = len(work)
    if workers > 1:
        if verbose:
            print(f"  Transcribing {total} segments across {workers} threads...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            transcribed = list(pool.map(_transcribe, work))
    else:
        transcribed = []
        for j, item in enumerate(work, 1):
            if verbose:
                print(f"  Segment {j}/{total}...")
            transcribed.append(_transcribe(item))

    if verbose:
        print(f"Transcription done in {int(time.time() - t0)}s.")

    # Rebuild per-speaker text lists from results
    speaker_texts: dict[str, list] = {tag: [] for tag in speakers}
    for tag, start, end, text in transcribed:
        speaker_texts[tag].append([start, end, text])

    # Assemble final output in diarization order
    result: list[dict] = []
    for start, end, speaker_name in ordered:
        for tag, texts in speaker_texts.items():
            if speaker_map[tag] == speaker_name:
                for seg_start, seg_end, text in texts:
                    if seg_start == start and seg_end == end:
                        result.append({
                            "file_name": file,
                            "start_time": start,
                            "end_time": end,
                            "text": text,
                            "speaker": speaker_name,
                            "model_used": model_label,
                            "language_detected": language or "auto",
                        })
    return result


def _run(
    file: str,
    diarizer: BaseDiarizer,
    recognizer: BaseRecognizer | None,
    asr_model: BaseASR,
    language: str | None,
    voices_folder: str | None,
    log_folder: str,
    output_format: str,
    verbose: bool,
    srt: bool,
    workers: int,
) -> list[dict]:
    file = _preprocess(file, verbose)
    segments = _process_audio(
        file, diarizer, recognizer, asr_model,
        language, voices_folder, verbose, workers,
    )

    write_log_file(segments, log_folder, file, language, output_format, srt=srt)
    return segments


class Pipeline:
    """
    Speech pipeline: diarization → (optional speaker recognition) → ASR.

    Diarization always runs on the full audio in a single pass to preserve
    global speaker context.  Transcription of the resulting segments is
    parallelised across ``workers`` threads.

    All provider-specific settings belong on the component instances.

    Parameters
    ----------
    diarization_model : BaseDiarizer
        e.g. ``PyAnnoteDiarizer(access_token="hf_...", max_speakers=4)``
    asr_model : BaseASR
        e.g. ``FasterWhisperASR("turbo", beam_size=5)``
    speaker_recognition_model : BaseRecognizer | None
        Recognizer instance, or ``None`` to skip.
    language : str | None
        BCP-47 language code, or ``None`` for automatic detection.
    voices_folder : str | None
        Root directory of per-speaker reference recordings.
    log_folder : str
        Output directory for transcript files (created if absent).
    output_format : str
        ``"txt"``, ``"json"``, or ``"both"``.
    verbose : bool
        Print per-segment progress and stage timings.
    srt : bool
        Also write an SRT subtitle file.
    workers : int | None
        Number of threads for parallel transcription.
        ``None``  → ``max(1, cpu_count - 1)``
        ``1``     → sequential transcription
        ``N > 1`` → N threads transcribe segments concurrently.
        Diarization is never parallelised regardless of this value.

    Example
    -------
    ::

        pipeline = Pipeline(
            diarization_model=PyAnnoteDiarizer(access_token="hf_...", max_speakers=4),
            asr_model=FasterWhisperASR("turbo", beam_size=5),
            language=None,
            log_folder="logs/",
            output_format="both",
            workers=None,
        )

        segments = pipeline.run("interview.wav")
        batched  = pipeline.run(["call1.wav", "call2.wav"])
    """

    def __init__(
        self,
        diarization_model: BaseDiarizer,
        asr_model: BaseASR,
        speaker_recognition_model: BaseRecognizer | None = None,
        language: str | None = None,
        voices_folder: str | None = None,
        log_folder: str = "logs",
        output_format: str = "both",
        verbose: bool = False,
        srt: bool = False,
        workers: int | None = None,
    ):
        if not isinstance(diarization_model, BaseDiarizer):
            raise TypeError("diarization_model must be a BaseDiarizer instance.")
        if not isinstance(asr_model, BaseASR):
            raise TypeError("asr_model must be a BaseASR instance.")
        if speaker_recognition_model is not None and not isinstance(
            speaker_recognition_model, BaseRecognizer
        ):
            raise TypeError(
                "speaker_recognition_model must be a BaseRecognizer instance or None."
            )

        self._diarizer = diarization_model
        self._recognizer = speaker_recognition_model
        self._asr = asr_model
        self.language = language
        self.voices_folder = voices_folder
        self.log_folder = log_folder
        self.output_format = output_format
        self.verbose = verbose
        self.srt = srt
        self.workers = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers

    def run(self, file: str | list[str]) -> list[dict] | list[list[dict]]:
        """
        Transcribe one file or a batch of files.

        Parameters
        ----------
        file : str | list[str]
            Path(s) to audio files.  Non-WAV formats are converted automatically.

        Returns
        -------
        list[dict]
            Single file: time-ordered segment dicts with keys
            ``file_name``, ``start_time``, ``end_time``, ``text``,
            ``speaker``, ``model_used``, ``language_detected``
        list[list[dict]]
            Batch: one inner list per input file, in input order.
        """
        files = file if isinstance(file, list) else [file]
        results = []
        total = len(files)

        for idx, f in enumerate(files, 1):
            if total > 1:
                print(f"\n[File {idx}/{total}] {f}")
            results.append(_run(
                file=f,
                diarizer=self._diarizer,
                recognizer=self._recognizer,
                asr_model=self._asr,
                language=self.language,
                voices_folder=self.voices_folder,
                log_folder=self.log_folder,
                output_format=self.output_format,
                verbose=self.verbose,
                srt=self.srt,
                workers=self.workers,
            ))

        return results[0] if len(files) == 1 else results
