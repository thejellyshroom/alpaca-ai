from setuptools import setup, find_namespace_packages

setup(
    name="alpaca-ai",
    version="0.1.0",
    packages=find_namespace_packages(include=["src.*"]),
    package_dir={"": "."},
    install_requires=[
        "python-dotenv==1.0.1",
        "ollama==0.4.7",
        "transformers==4.51.3",
        "tiktoken==0.9.0",
        "nltk==3.8.1",
        "rouge-score==0.1.2",
        "json_repair==0.41.1",
        "tenacity==8.5.0",
        "faster-whisper==1.1.1",
        "SpeechRecognition==3.14.1",
        "kokoro==0.7.16",
        "pyaudio==0.2.14",
        "sounddevice==0.5.1",
        "soundfile==0.13.1",
        "silero-vad==5.1.2",
        "numpy==1.26.4",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        "pipmaster==0.5.4"
    ],
    python_requires=">=3.9",
) 