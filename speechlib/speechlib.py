from .core_analysis import (core_analysis)
from .re_encode import (re_encode)
from .convert_to_mono import (convert_to_mono)
from .convert_to_wav import (convert_to_wav)

class Transcriptor:

    def __init__(self, file, log_folder, language, modelSize, ACCESS_TOKEN, voices_folder=None, quantization=False):
        '''transcribe a wav file 
        
            arguments:

            file: name of wav file with extension ex: file.wav

            log_folder: name of folder where transcript will be stored

            language: language of wav file

            modelSize: tiny, small, medium, large, large-v1, large-v2, large-v3 (bigger model is more accurate but slow!!)

            voices_folder: folder containing subfolders named after each speaker with speaker voice samples in them. This will be used for speaker recognition

            quantization: whether to use int8 quantization or not (default=False)

            see documentation: https://github.com/Navodplayer1/speechlib
        
            
        supported languages:  
        #### Afrikaans
        "af",
        #### Amharic
        "am",
        #### Arabic
        "ar",
        #### Assamese
        "as",
        #### Azerbaijani
        "az",
        #### Bashkir
        "ba",
        #### Belarusian
        "be",
        #### Bulgarian
        "bg",
        #### Bengali
        "bn",
        #### Tibetan
        "bo",
        #### Breton
        "br",
        #### Bosnian
        "bs",
        #### Catalan
        "ca",
        #### Czech
        "cs",
        #### Welsh
        "cy",
        #### Danish
        "da",
        #### German
        "de",
        #### Greek
        "el",
        #### English
        "en",
        #### Spanish
        "es",
        #### Estonian
        "et",
        #### Basque
        "eu",
        #### Persian
        "fa",
        #### Finnish
        "fi",
        #### Faroese
        "fo",
        #### French
        "fr",
        #### Galician
        "gl",
        #### Gujarati
        "gu",
        #### Hausa
        "ha",
        #### Hawaiian
        "haw",
        #### Hebrew
        "he",
        #### Hindi
        "hi",
        #### Croatian
        "hr",
        #### Haitian
        "ht",
        #### Hungarian
        "hu",
        #### Armenian
        "hy",
        #### Indonesian
        "id",
        #### Icelandic
        "is",
        #### Italian
        "it",
        #### Japanese
        "ja",
        #### Javanese
        "jw",
        #### Georgian
        "ka",
        #### Kazakh
        "kk",
        #### Khmer
        "km",
        #### Kannada
        "kn",
        #### Korean
        "ko",
        #### Latin
        "la",
        #### Luxembourgish
        "lb",
        #### Lingala
        "ln",
        #### Lao
        "lo",
        #### Lithuanian
        "lt",
        #### Latvian
        "lv",
        #### Malagasy
        "mg",
        #### Maori
        "mi",
        #### Macedonian
        "mk",
        #### Malayalam
        "ml",
        #### Mongolian
        "mn",
        #### Marathi
        "mr",
        #### Malay
        "ms",
        #### Maltese
        "mt",
        #### Burmese
        "my",
        #### Nepali
        "ne",
        #### Dutch
        "nl",
        #### Norwegian Nynorsk
        "nn",
        #### Norwegian
        "no",
        #### Occitan
        "oc",
        #### Punjabi
        "pa",
        #### Polish
        "pl",
        #### Pashto
        "ps",
        #### Portuguese
        "pt",
        #### Romanian
        "ro",
        #### Russian
        "ru",
        #### Sanskrit
        "sa",
        #### Sindhi
        "sd",
        #### Sinhalese
        "si",
        #### Slovak
        "sk",
        #### Slovenian
        "sl",
        #### Shona
        "sn",
        #### Somali
        "so",
        #### Albanian
        "sq",
        #### Serbian
        "sr",
        #### Sundanese
        "su",
        #### Swedish
        "sv",
        #### Swahili
        "sw",
        #### Tamil
        "ta",
        #### Telugu
        "te",
        #### Tajik
        "tg",
        #### Thai
        "th",
        #### Turkmen
        "tk",
        #### Tagalog
        "tl",
        #### Turkish
        "tr",
        #### Tatar
        "tt",
        #### Ukrainian
        "uk",
        #### Urdu
        "ur",
        #### Uzbek
        "uz",
        #### Vietnamese
        "vi",
        #### Yiddish
        "yi",
        #### Yoruba
        "yo",
        #### Chinese
        "zh",
        #### Cantonese
        "yue",
        '''
        self.file = file
        self.voices_folder = voices_folder
        self.language = language
        self.log_folder = log_folder
        self.modelSize = modelSize
        self.quantization = quantization
        self.ACCESS_TOKEN = ACCESS_TOKEN

    def whisper(self):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "whisper", self.quantization)
        return res
    
    def faster_whisper(self):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "faster-whisper", self.quantization)
        return res

    def custom_whisper(self, custom_model_path):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "custom", self.quantization, custom_model_path)
        return res
    
    def huggingface_model(self, hf_model_id):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "huggingface", self.quantization, None, hf_model_id)
        return res

class PreProcessor:
    '''
    class for preprocessing audio files.

    methods:

    re_encode(file) -> re-encode file to 16-bit PCM encoding  

    convert_to_mono(file) -> convert file from stereo to mono  

    mp3_to_wav(file) -> convert mp3 file to wav format  

    '''

    def re_encode(self, file):
        re_encode(file)
    
    def convert_to_mono(self, file):
        convert_to_mono(file)

    def convert_to_wav(self, file):
        path = convert_to_wav(file)
        return path
