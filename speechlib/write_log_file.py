import os
from datetime import datetime

def write_log_file(common_segments, log_folder, file_name, language):

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    #---------------------log file part-------------------------
        
    current_time = datetime.now().strftime('%H%M%S')

    file_name = os.path.splitext(os.path.basename(file_name))[0]

    log_file = log_folder + "/" + file_name + "_" + current_time + "_" + language + ".txt"
    
    lf=open(log_file,"wb")

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
