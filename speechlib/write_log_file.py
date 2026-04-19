import os
import json
from datetime import datetime


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60000) % 60
    h = total_ms // 3600000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_log_file(common_segments, log_folder, file_name, language, output_format="both", srt: bool = False):

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    #---------------------log file part-------------------------
        
    current_time = datetime.now().strftime('%H%M%S')

    base_name = os.path.splitext(os.path.basename(file_name))[0]
    lang_str = str(language) if language else "auto"

    log_file_txt = log_folder + "/" + base_name + "_" + current_time + "_" + lang_str + ".txt"
    log_file_json = log_folder + "/" + base_name + "_" + current_time + "_" + lang_str + ".json"

    entry = ""
    
    for segment in common_segments:
        start = segment["start_time"]
        end = segment["end_time"]
        text = segment["text"]
        speaker = segment["speaker"]
        
        if text != "" and text != None:
            entry += f"{speaker} ({start} : {end}) : {text}\n"
        
    saved_files = []
    
    if output_format in ["txt", "both"]:
        with open(log_file_txt, "wb") as lf:
            lf.write(bytes(entry.encode('utf-8')))
        saved_files.append(log_file_txt)

    if output_format in ["json", "both"]:
        # JSON log file
        json_data = {
            "file_name": file_name,
            "language_detected": common_segments[0].get("language_detected", lang_str) if common_segments else lang_str,
            "model_used": common_segments[0].get("model_used", "unknown") if common_segments else "unknown",
            "segments": [
                {
                    "start_time": seg.get("start_time"),
                    "end_time": seg.get("end_time"),
                    "text": seg.get("text"),
                    "speaker": seg.get("speaker"),
                }
                for seg in common_segments if seg.get("text") != ""
            ]
        }
        
        with open(log_file_json, "w", encoding='utf-8') as jf:
            json.dump(json_data, jf, ensure_ascii=False, indent=4)
        saved_files.append(log_file_json)
        
    if srt:
        log_file_srt = log_folder + "/" + base_name + "_" + current_time + "_" + lang_str + ".srt"
        srt_entries = []
        for idx, segment in enumerate(common_segments, 1):
            start = _srt_timestamp(segment["start_time"])
            end = _srt_timestamp(segment["end_time"])
            text = segment.get("text", "")
            speaker = segment.get("speaker", "")
            if text:
                srt_entries.append(f"{idx}\n{start} --> {end}\n{speaker}: {text}\n")
        with open(log_file_srt, "w", encoding="utf-8") as sf:
            sf.write("\n".join(srt_entries))
        saved_files.append(log_file_srt)

    if saved_files:
        print(f"Log files saved: {', '.join(saved_files)}")

    # -------------------------log file end-------------------------
