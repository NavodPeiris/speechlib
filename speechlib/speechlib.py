import os
from .core_analysis import (core_analysis)
from .re_encode import (re_encode)
from .convert_to_mono import (convert_to_mono)
from .convert_to_wav import (convert_to_wav)

class Transcriptor:

    def __init__(self, file, log_folder="logs", language=None, modelSize="tiny", ACCESS_TOKEN=None, voices_folder=None, quantization=False, output_format="both", verbose=False, **kwargs):
        '''
        transcribe a wav file or a list of wav files
        
        arguments:

        file: name of wav file with extension ex: file.wav, or a list of such files

        log_folder: name of folder where transcript will be stored

        language: language of wav file (default=None for auto-detect)

        modelSize: tiny, small, medium, large, large-v1, large-v2, large-v3, turbo (bigger model is more accurate but slow!!)

        ACCESS_TOKEN: huggingface access token (defaults to HUGGINGFACE_ACCESS_TOKEN from env)

        voices_folder: folder containing subfolders named after each speaker with speaker voice samples in them. This will be used for speaker recognition

        quantization: whether to use int8 quantization or not (default=False)
        
        output_format: "txt", "json", or "both" (default="both")

        kwargs: any additional parameters (like beam_size, min_speakers, max_speakers, etc.)

        see documentation: https://github.com/NavodPeiris/speechlib
        
            
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
        self.file = file if isinstance(file, list) else [file]
        self.voices_folder = voices_folder
        self.language = language
        self.log_folder = log_folder
        self.modelSize = modelSize
        self.quantization = quantization
        self.output_format = output_format
        self.verbose = verbose
        self.ACCESS_TOKEN = ACCESS_TOKEN or os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
        self.kwargs = kwargs

    def _process_batch(self, model_type, custom_model_path=None, hf_model_id=None, aai_api_key=None):
        results = []
        total_files = len(self.file)
        for idx, f in enumerate(self.file, 1):
            if total_files > 1:
                print(f"\n[File {idx}/{total_files}] Starting processing for {f} ...")
            else:
                print(f"\nStarting processing for {f} ...")
            try:
                res = core_analysis(f, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, model_type, self.quantization, custom_model_path, hf_model_id, aai_api_key, self.output_format, verbose=self.verbose, **self.kwargs)
                results.append(res)
            except Exception as e:
                if total_files > 1:
                    print(f"[File {idx}/{total_files}] ERROR processing {f}: {e}")
                else:
                    print(f"ERROR processing {f}: {e}")
        # If single file, return the single result for backwards compatibility
        if len(self.file) == 1 and len(results) == 1:
            return results[0]
        return results

    def whisper(self):
        return self._process_batch("whisper")
    
    def faster_whisper(self):
        return self._process_batch("faster-whisper")

    def custom_whisper(self, custom_model_path):
        return self._process_batch("custom", custom_model_path=custom_model_path)
    
    def huggingface_model(self, hf_model_id):
        return self._process_batch("huggingface", hf_model_id=hf_model_id)
    
    def assemby_ai_model(self, aai_api_key):
        return self._process_batch("assemblyAI", aai_api_key=aai_api_key)

class PreProcessor:
    '''
    class for preprocessing audio files.

    methods:

    re_encode(file) -> re-encode file to 16-bit PCM encoding  

    convert_to_mono(file) -> convert file from stereo to mono  

    mp3_to_wav(file) -> convert mp3 file to wav format  

    '''

    def re_encode(self, file, verbose=False):
        re_encode(file, verbose=verbose)
    
    def convert_to_mono(self, file, verbose=False):
        convert_to_mono(file, verbose=verbose)

    def convert_to_wav(self, file, verbose=False):
        path = convert_to_wav(file, verbose=verbose)
        return path
