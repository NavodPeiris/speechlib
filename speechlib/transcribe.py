from .whisper_sinhala import (whisper_sinhala)
from .whisper_medium import (whisper_medium)
from .whisper_large import (whisper_large)
from .whisper_tiny import (whisper_tiny)

def transcribe(file, language, modelSize):

    if language == "sinhala" or language == "Sinhala":
        res = whisper_sinhala(file)
        return res
    elif modelSize == "medium":
        res = whisper_medium(file, language)
        return res
    elif modelSize == "large":
        res = whisper_large(file, language)
        return res
    elif modelSize == "tiny":
        res = whisper_tiny(file, language)
        return res
    else:
        raise Exception("only tiny, medium, large models are available. If you use Sinhala language, use tiny model")

