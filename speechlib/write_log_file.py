import os
import json
from datetime import datetime

def write_log_file(common_segments, log_folder, file_name, language, output_format="both"):

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
            "segments": common_segments
        }
        
        with open(log_file_json, "w", encoding='utf-8') as jf:
            json.dump(json_data, jf, ensure_ascii=False, indent=4)
        saved_files.append(log_file_json)
        
    if saved_files:
        print(f"Log files saved: {', '.join(saved_files)}")

    # -------------------------log file end-------------------------
