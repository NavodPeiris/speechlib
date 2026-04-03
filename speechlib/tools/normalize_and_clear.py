#!/usr/bin/env python3
"""
Normalize and clear (enhance) audio files.
Reuses speechlib pipeline infrastructure.

Usage:
    python -m speechlib.tools.normalize_and_clear "path/to/audio.m4a"
    python -m speechlib.tools.normalize_and_clear "path/to/*.m4a"
    python -m speechlib.tools.normalize_and_clear "path/to/*.m4a" --skip-clear
"""

import argparse
import glob
import shutil
import time
from datetime import datetime
from pathlib import Path
import soundfile as sf

from speechlib.audio_state import AudioState
from speechlib.convert_to_wav import convert_to_wav
from speechlib.convert_to_mono import convert_to_mono
from speechlib.re_encode import re_encode
from speechlib.resample_to_16k import resample_to_16k
from speechlib.loudnorm import loudnorm
from speechlib.enhance_audio import enhance_audio


def get_audio_duration(path: Path) -> float:
    """Get audio duration in seconds."""
    try:
        info = sf.info(str(path))
        return info.duration
    except Exception:
        return 0.0


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS."""
    if seconds <= 0:
        return "?:??"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def normalize_and_clear(
    input_path: str | Path,
    skip_clear: bool = False,
) -> Path:
    """Normalize (loudnorm) and clear (enhance) an audio file.

    Uses same output structure as speechlib pipeline:
    - artifacts_dir: .<stem>/ alongside source file
    - output: artifacts_dir/processed.wav

    Args:
        input_path: Path to input audio file
        skip_clear: Skip ClearVoice enhancement step

    Returns:
        Path to the processed audio file
    """
    input_path = Path(input_path)
    start_time = time.time()

    duration = get_audio_duration(input_path)
    duration_str = format_duration(duration)

    state = AudioState(source_path=input_path, working_path=input_path)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    cached_processed = state.artifacts_dir / "processed.wav"
    if cached_processed.exists():
        print(f"[cached] {cached_processed}")
        return cached_processed

    print(f"\n{'=' * 60}")
    print(f"  {input_path.name}")
    print(
        f"  Duration: {duration_str} | Started: {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"{'=' * 60}")
    print(f"  Output: {state.artifacts_dir.name}/")

    steps = [
        ("WAV", convert_to_wav),
        ("Mono", convert_to_mono),
        ("16-bit", re_encode),
        ("16kHz", resample_to_16k),
        ("Normalize", loudnorm),
    ]

    for i, (name, func) in enumerate(steps, 1):
        step_start = time.time()
        state = func(state)
        step_elapsed = time.time() - step_start
        print(f"  [{i}/{len(steps)}] {name:<12} ({step_elapsed:.1f}s)")

    if not skip_clear:
        clear_start = time.time()
        print(f"  [{len(steps) + 1}/{len(steps) + 1}] Clear (MossFormer2)...")
        state = enhance_audio(state)
        clear_elapsed = time.time() - clear_start
        print(f"           Done ({clear_elapsed:.1f}s)")
    else:
        print(f"  [{len(steps) + 1}/{len(steps) + 1}] Clear (skipped)")

    shutil.copy(state.working_path, cached_processed)

    total_elapsed = time.time() - start_time
    print(
        f"\n  Completed in {format_duration(total_elapsed)} ({datetime.now().strftime('%H:%M:%S')})"
    )
    print(f"  Output: {cached_processed}")

    return cached_processed


def get_files(pattern: str) -> list[Path]:
    """Get files matching glob pattern."""
    pattern_path = Path(pattern)

    if pattern_path.exists() and pattern_path.is_file():
        return [pattern_path]

    if pattern_path.exists() and pattern_path.is_dir():
        return []

    parent = pattern_path.parent
    stem = pattern_path.name

    if parent.exists():
        files = sorted(parent.glob(stem))
        if files:
            return files

    files = sorted(Path.cwd().glob(pattern))
    if files:
        return files

    return []


def main():
    parser = argparse.ArgumentParser(description="Normalize and clear audio files")
    parser.add_argument(
        "input",
        nargs="+",
        help="Input audio file(s) or glob pattern (e.g., *.m4a)",
    )
    parser.add_argument(
        "--skip-clear",
        action="store_true",
        help="Skip ClearVoice enhancement",
    )

    args = parser.parse_args()

    files = []
    for pattern in args.input:
        matched = get_files(pattern)
        files.extend(matched)

    if not files:
        print(f"No files found matching: {args.input}")
        return

    print(f"{'=' * 60}")
    print(f"  Found {len(files)} file(s)")
    print(f"{'=' * 60}")

    success = 0
    failed = 0

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {file_path.name}")
        try:
            normalize_and_clear(file_path, args.skip_clear)
            success += 1
            print(f"  OK")
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Summary: {success} OK, {failed} failed of {len(files)} files")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
