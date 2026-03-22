import os
from datetime import datetime


def _format_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_log_file(common_segments, log_folder, file_name, language, output_format: str = "vtt"):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    current_time = datetime.now().strftime('%H%M%S')
    base_name = os.path.splitext(os.path.basename(file_name))[0]

    if output_format == "vtt":
        log_file = f"{log_folder}/{base_name}_{current_time}_{language}.vtt"
        entry = "WEBVTT\n\n"
        n = 0
        for segment in common_segments:
            start, end, text, speaker = segment[0], segment[1], segment[2], segment[3]
            if text:
                n += 1
                entry += f"{n}\n{_format_vtt(start)} --> {_format_vtt(end)}\n[{speaker}] {text}\n\n"
    else:
        log_file = f"{log_folder}/{base_name}_{current_time}_{language}.txt"
        entry = ""
        for segment in common_segments:
            start, end, text, speaker = segment[0], segment[1], segment[2], segment[3]
            if text:
                entry += f"{speaker} ({start} : {end}) : {text}\n"

    with open(log_file, "wb") as lf:
        lf.write(entry.encode('utf-8'))
