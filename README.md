supported languages:  

['english', 'chinese', 'german', 'spanish', 'russian', 'korean', 'french', 'japanese', 'portuguese', 'turkish', 'polish', 'catalan', 'dutch', 'arabic', 'swedish', 'italian', 'indonesian', 'hindi', 'finnish', 'vietnamese', 'hebrew', 'ukrainian', 'greek', 'malay', 'czech', 'romanian', 'danish', 'hungarian', 'tamil', 'norwegian', 'thai', 'urdu', 'croatian', 'bulgarian', 'lithuanian', 'latin', 'maori', 'malayalam', 'welsh', 'slovak', 'telugu', 'persian', 'latvian', 'bengali', 'serbian', 'azerbaijani', 'slovenian', 'kannada', 'estonian', 'macedonian', 'breton', 'basque', 'icelandic', 'armenian', 'nepali', 'mongolian', 'bosnian', 'kazakh', 'albanian', 'swahili', 'galician', 'marathi', 'punjabi', 'sinhala', 'khmer', 'shona', 'yoruba', 'somali', 'afrikaans', 'occitan', 'georgian', 'belarusian', 'tajik', 'sindhi', 'gujarati', 'amharic', 'yiddish', 'lao', 'uzbek', 'faroese', 'haitian creole', 'pashto', 'turkmen', 'nynorsk', 'maltese', 'sanskrit', 'luxembourgish', 'myanmar', 'tibetan', 'tagalog', 'malagasy', 'assamese', 'tatar', 'hawaiian', 'lingala', 'hausa', 'bashkir', 'javanese', 'sundanese', 'burmese', 'valencian', 'flemish', 'haitian', 'letzeburgesch', 'pushto', 'panjabi', 'moldavian', 'moldovan', 'sinhalese', 'castilian']

for building setup:
    pip install setuptools
    pip install wheel

on root:
    python setup.py sdist bdist_wheel

for publishing:
    pip install twine

for install locally for testing:
    pip install dist/speechlib-1.0.0-py3-none-any.whl

finally run:
    twine upload dist/*

    fill as follows:
        username: __token__
        password: pypi-AgEIcHlwaS5vcmcCJDM1MjQ3NmE2LTUzMmItNDU1MS04ZmZjLWMzY2YyMDlkNWQyMwACKlszLCI2YmYwNjMzNy04YWVlLTQ4M2EtOTA4OS03ZWEzYjA1ZjZkMGUiXQAABiBWbNaxjL0FrtwoysrTXCcaBzFyoPCaq5ldblFbylT_bQ