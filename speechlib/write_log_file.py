import os
from datetime import datetime

def write_log_file(common_segments, log_folder):

    file_name = "output"
    current_datetime = datetime.now().strftime("%Y-%m-%d")

    #---------------------log file part-------------------------

    log_file = file_name + "_" + current_datetime + ".txt"

    lf=open(os.path.join(log_folder, log_file),"wb")

    entry = ""
    
    for segment in common_segments:
        start = segment[0]
        end = segment[1]
        text = segment[2]
        speaker = segment[3]
        
        entry += f"{speaker} ({start} : {end}) : {text}\n"
        
    lf.write(bytes(entry.encode('utf-8')))      
    lf.close()

    # -------------------------log file end-------------------------
