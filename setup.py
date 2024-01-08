from setuptools import find_packages, setup

with open("library.md", "r") as f:
    long_description = f.read()

setup(
    name="speechlib",
    version="1.0.1",
    description="speechlib is a library that can do speaker diarization, transcription and speaker recognition on an audio file to create transcripts with actual speaker names",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Navodplayer1/speechlib",
    author="Navod Peiris",
    author_email="navodpeiris1234@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["transformers", "torch", "torchaudio", "pydub", "pyannote.audio", "speechbrain"],
    python_requires=">=3.7",
)