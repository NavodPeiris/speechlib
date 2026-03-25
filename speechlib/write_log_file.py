from pathlib import Path

from .vtt_utils import seconds_to_vtt_ts as _format_vtt


def write_log_file(common_segments, log_folder, file_name, language, output_format: str = "vtt"):
    artifacts_dir = Path(file_name).parent

    if output_format == "vtt":
        entry = "WEBVTT\n\n"
        n = 0
        for segment in common_segments:
            start, end, text, speaker = segment[0], segment[1], segment[2], segment[3]
            if text:
                n += 1
                entry += f"{n}\n{_format_vtt(start)} --> {_format_vtt(end)}\n[{speaker}] {text}\n\n"
        out_path = artifacts_dir / f"transcript_{language}.vtt"
    else:
        entry = ""
        for segment in common_segments:
            start, end, text, speaker = segment[0], segment[1], segment[2], segment[3]
            if text:
                entry += f"{speaker} ({start} : {end}) : {text}\n"
        out_path = artifacts_dir / f"transcript_{language}.txt"

    out_path.write_bytes(entry.encode("utf-8"))
