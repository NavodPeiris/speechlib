<p align="center">
  <img src="speechlib.png" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/github/license/Navodplayer1/speechlib"></a>
    <a href="https://github.com/Navodplayer1/speechlib/releases"><img src="https://img.shields.io/github/v/release/Navodplayer1/speechlib?color=ffa"></a>
    <a href="support os"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href="https://github.com/Navodplayer1/speechlib/graphs/contributors"><img src="https://img.shields.io/github/contributors/Navodplayer1/speechlib?color=9ea"></a>
    <a href="https://github.com/Navodplayer1/speechlib/issues"><img src="https://img.shields.io/github/issues/Navodplayer1/speechlib?color=9cc"></a>
    <a href="https://github.com/Navodplayer1/speechlib/stargazers"><img src="https://img.shields.io/github/stars/Navodplayer1/speechlib?color=ccf"></a>
    <a href="https://pypi.org/project/speechlib/"><img src="https://static.pepy.tech/badge/speechlib"></a>
    
</p>


### Requirements

* Python 3.8 or greater

### GPU execution

GPU execution needs CUDA 11.  

GPU execution requires the following NVIDIA libraries to be installed:

* [cuBLAS for CUDA 11](https://developer.nvidia.com/cublas)
* [cuDNN 8 for CUDA 11](https://developer.nvidia.com/cudnn)

There are multiple ways to install these libraries. The recommended way is described in the official NVIDIA documentation, but we also suggest other installation methods below.

### Google Colab:

on google colab run this to install CUDA dependencies:
```
!apt install libcublas11
```

You can see this example [notebook]()

### installation:
```
pip install speechlib
```

This library does speaker diarization, speaker recognition, and transcription on a single wav file to provide a transcript with actual speaker names. This library will also return an array containing result information. ⚙ 

This library contains following audio preprocessing functions:

1. convert mp3 to wav

2. convert stereo wav file to mono

3. re-encode the wav file to have 16-bit PCM encoding

Transcriptor method takes 6 arguments. 

1. file to transcribe

2. log_folder to store transcription

3. language used for transcribing (language code is used)

4. model size ("tiny", "small", "medium", "large", "large-v1", "large-v2", "large-v3")

5. voices_folder (contains speaker voice samples for speaker recognition)

6. quantization: this determine whether to use int8 quantization or not. Quantization may speed up the process but lower the accuracy.

voices_folder should contain subfolders named with speaker names. Each subfolder belongs to a speaker and it can contain many voice samples. This will be used for speaker recognition to identify the speaker.

if voices_folder is not provided then speaker tags will be arbitrary.

log_folder is to store the final transcript as a text file.

transcript will also indicate the timeframe in seconds where each speaker speaks.

### Transcription example:

```
from speechlib import Transcriptor

file = "obama_zach.wav"
voices_folder = "voices"
language = "en"
log_folder = "logs"
modelSize = "medium"
quantization = False   # setting this 'True' may speed up the process but lower the accuracy

transcriptor = Transcriptor(file, log_folder, language, modelSize, voices_folder, quantization)

res = transcriptor.transcribe()

res --> [["start", "end", "text", "speaker"], ["start", "end", "text", "speaker"]...]
```

start: starting time of speech in seconds  
end: ending time of speech in seconds  
text: transcribed text for speech during start and end  
speaker: speaker of the text 

voices_folder structure:  
```
voices_folder    
|---> person1      
|        |---> sample1.wav   
|        |---> sample2.wav     
|                ...
|
|---> person2  
|        |---> sample1.wav  
|        |---> sample2.wav   
|                ...
|--> ...  
```

supported language codes:  

```
"af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is","it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn","mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk","sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz","vi", "yi", "yo", "zh", "yue"
```

supported language names:

```
"Afrikaans", "Amharic", "Arabic", "Assamese", "Azerbaijani", "Bashkir", "Belarusian", "Bulgarian", "Bengali","Tibetan", "Breton", "Bosnian", "Catalan", "Czech", "Welsh", "Danish", "German", "Greek", "English", "Spanish","Estonian", "Basque", "Persian", "Finnish", "Faroese", "French", "Galician", "Gujarati", "Hausa", "Hawaiian","Hebrew", "Hindi", "Croatian", "Haitian", "Hungarian", "Armenian", "Indonesian", "Icelandic", "Italian", "Japanese","Javanese", "Georgian", "Kazakh", "Khmer", "Kannada", "Korean", "Latin", "Luxembourgish", "Lingala", "Lao","Lithuanian", "Latvian", "Malagasy", "Maori", "Macedonian", "Malayalam", "Mongolian", "Marathi", "Malay", "Maltese","Burmese", "Nepali", "Dutch", "Norwegian Nynorsk", "Norwegian", "Occitan", "Punjabi", "Polish", "Pashto","Portuguese", "Romanian", "Russian", "Sanskrit", "Sindhi", "Sinhalese", "Slovak", "Slovenian", "Shona", "Somali","Albanian", "Serbian", "Sundanese", "Swedish", "Swahili", "Tamil", "Telugu", "Tajik", "Thai", "Turkmen", "Tagalog","Turkish", "Tatar", "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Yiddish", "Yoruba", "Chinese", "Cantonese",
```

### Audio preprocessing example:

```
from speechlib import PreProcessor

file = "obama1.mp3"

# convert mp3 to wav
wav_file = PreProcessor.convert_to_wav(file)   

# convert wav file from stereo to mono
PreProcessor.convert_to_mono(wav_file)

# re-encode wav file to have 16-bit PCM encoding
PreProcessor.re_encode(wav_file)
```

This library uses following huggingface models:

#### https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
#### https://huggingface.co/Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2
#### https://huggingface.co/pyannote/speaker-diarization