This library do speaker diarization, speaker recognition, transcription on a single wav file to provide a transcript with actual speaker names. This library will also return an array containing result information.

Transcriptor takes 4 arguments. file to transcribe, log_folder, language used for transcribing, voices folder

voices_folder should contain subfolders named with speaker names and their voice samples. This will be used for speaker recognition to identify speaker.

if voice_folder is not provided then speaker tags will be arbitrary.

log_folder is to store final transcript as a text file.

example:

```
from speechlib import Transcriptor

file = "obama.wav"
voice_folder = "voices"
language = "english"
log_folder = "logs"

transcriptor = Transcriptor(file, log_folder, language, voice_folder)

res = transcriptor.transcribe()

print(res)

--> [["start", "end", "text", "speaker"], ["start", "end", "text", "speaker"]...]
```

start: starting time of speech  
end: ending time of speech  
text: transcribed text for speech   during start and end  
speaker: speaker of the text 

This library uses following huggingface models:

#### https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
#### https://huggingface.co/Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2
#### https://huggingface.co/openai/whisper-medium
#### https://huggingface.co/pyannote/speaker-diarization