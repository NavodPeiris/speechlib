from .core_analysis import (core_analysis)
from .re_encode import (re_encode)
from .convert_to_mono import (convert_to_mono)
from .mp3_to_wav import (mp3_to_wav)

class Transcriptor:
    '''transcribe a wav file 
    
        arguments:

        file: name of wav file with extension ex: file.wav

        log_folder: name of folder where transcript will be stored

        language: language of wav file

        modelSize: tiny, medium, large (bigger model is more accurate but slow!!)

        voices_folder: folder containing subfolders named after each speaker with speaker voice samples in them. This will be used for speaker recognition

        see documentation: https://github.com/Navodplayer1/speechlib
    '''

    def __init__(self, file, log_folder, language, modelSize, voices_folder=None):
        '''
        supported languages:  
        ['english', 
        'chinese', 
        'german', 
        'spanish', 
        'russian', 
        'korean', 
        'french', '
        japanese', 
        'portuguese', 
        'turkish', 
        'polish', 
        'catalan', 
        'dutch', 
        'arabic', 
        'swedish', 
        'italian', 
        'indonesian', 
        'hindi', 
        'finnish', 
        'vietnamese', 
        'hebrew', 
        'ukrainian', 
        'greek', 
        'malay', 
        'czech', 
        'romanian', 
        'danish', 
        'hungarian', 
        'tamil', 
        'norwegian', 
        'thai', 
        'urdu', 
        'croatian', 
        'bulgarian', 
        'lithuanian', 
        'latin', 
        'maori', 
        'malayalam', 
        'welsh', 
        'slovak', 
        'telugu', 
        'persian', 
        'latvian', 
        'bengali', 
        'serbian', 
        'azerbaijani', 
        'slovenian', 
        'kannada', 
        'estonian', 
        'macedonian', 
        'breton', 
        'basque', 
        'icelandic', 
        'armenian', 
        'nepali', 
        'mongolian', 
        'bosnian', 
        'kazakh', 
        'albanian', 
        'swahili', 
        'galician', 
        'marathi', 
        'punjabi', 
        'sinhala', 
        'khmer', 
        'shona', 
        'yoruba', 
        'somali', 
        'afrikaans', 
        'occitan', 
        'georgian', 
        'belarusian', 
        'tajik', 
        'sindhi', 
        'gujarati', 
        'amharic', 
        'yiddish', 
        'lao', 
        'uzbek', 
        'faroese', 
        'haitian creole', 
        'pashto', 
        'turkmen', 
        'nynorsk', 
        'maltese', 
        'sanskrit', 
        'luxembourgish', 
        'myanmar', 
        'tibetan', 
        'tagalog', 
        'malagasy', 
        'assamese', 
        'tatar', 
        'hawaiian', 
        'lingala', 
        'hausa', 
        'bashkir', 
        'javanese', 
        'sundanese', 
        'burmese', 
        'valencian', 
        'flemish', 
        'haitian', 
        'letzeburgesch', 
        'pushto', 
        'panjabi', 
        'moldavian', 
        'moldovan', 
        'castilian']
        '''
        self.file = file
        self.voices_folder = voices_folder
        self.language = language
        self.log_folder = log_folder
        self.modelSize = modelSize

    def transcribe(self):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize)
        return res

class PreProcessor:
    '''
    class for preprocessing audio files.

    methods:

    re_encode(file) -> re-encode file to 16-bit PCM encoding  

    convert_to_mono(file) -> convert file from stereo to mono  

    mp3_to_wav(file) -> convert mp3 file to wav format  

    '''

    def re_encode(file):
        re_encode(file)
    
    def convert_to_mono(file):
        convert_to_mono(file)

    def mp3_to_wav(file):
        mp3_to_wav(file)
