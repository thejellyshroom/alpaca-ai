# Alpaca Voice Assistant

A locally-runnable AI voice assistant leveraging powerful open-source models for transcription, language understanding, and speech synthesis. Designed for responsiveness with streaming capabilities and interruptible playback.

## Features

- **Local First:** Runs entirely on your machine (requires Ollama).
- **Modular Architecture:** Core components (audio, STT, LLM, TTS, conversation, config) are separated for better maintainability.
- **Streaming TTS:** Assistant starts speaking as soon as the first part of the response is generated.
- **Interruptible Playback:** Uses Silero VAD and RMS energy detection to allow the user to interrupt the assistant mid-speech.
- **RAG Integration:** Utilizes Retrieval-Augmented Generation via ChromaDB and Ollama embeddings to answer questions based on imported documents.
- **Configurable:** Easily configure models, presets, VAD sensitivity, and other parameters via JSON files.
- **Models Used (Defaults):**
  - **ASR:** `faster-whisper` (`Systran/faster-whisper-small`)
  - **LLM:** `ollama` (`gemma3:4b`)
  - **TTS:** Kokoro TTS
  - **Embedding (for RAG):** `nomic-embed-text` (via Ollama)
  - **VAD:** `silero-vad`

## Architecture

- `main.py`: Entry point, handles initialization and the main execution loop.
- `src/core/`: Contains the core orchestration classes:
  - `alpaca.py`: Main container class (`Alpaca`, formerly `VoiceAssistant`).
  - `alpaca_interaction.py`: Handles the logic for a single interaction turn (`InteractionHandler`).
  - `component_manager.py`: Manages the lifecycle (loading, access, cleanup) of utility handlers.
  - `conversation_manager.py`: Manages the conversation history.
  - `config_loader.py`: Handles command-line argument parsing and loading JSON configurations.
- `src/utils/`: Contains specialized handlers for specific tasks:
  - `audio_handler.py`: Coordinates audio input (listening) and output (playback, interruption).
  - `audio_player.py`: Handles audio playback queue and thread.
  - `interrupt_detector.py`: Handles VAD and RMS energy detection for interruptions.
  - `stt_handler.py`: Interface for the chosen speech-to-text model (e.g., `Transcriber`).
  - `llm_handler.py`: Interface for the chosen large language model (e.g., Ollama).
  - `tts_handler.py`: Interface for the chosen text-to-speech model.
  - `helper_functions.py`: Utility functions.
- `src/rag/`: Contains scripts related to RAG:
  - `importdocs.py`: Script to process and import documents into the ChromaDB vector store.
- `src/config/`: Contains JSON configuration files (`conf_asr.json`, `conf_tts.json`, `conf_llm.json`).

## Setup & Installation

**Prerequisites:**

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running.
- [ChromaDB](https://www.trychroma.com/) server running (for RAG functionality). By default, it's expected on `http://localhost:8000`.
- Microphone connected and configured.
- Speakers/Headphones.

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install Python dependencies:**
    - It's highly recommended to use a virtual environment:
      ```bash
      python -m venv venv
      source venv/bin/activate # On Windows use `venv\Scripts\activate`
      ```
    - Install required packages (You should create a `requirements.txt` file):
      ```bash
      # Example requirements (create a requirements.txt with exact versions)
      pip install -U torch torchaudio torchvision # Ensure compatibility!
      pip install -U ollama faster-whisper SpeechRecognition pyaudio sounddevice soundfile numpy
      pip install -U silero-vad onnxruntime # For VAD
      pip install -U chromadb-client # For RAG
      # Add any other specific dependencies (e.g., google-cloud-texttospeech)
      ```
      _Note: Verify PyTorch version compatibility with `torchvision` and potentially other libraries._
3.  **Download Ollama Models:**
    ```bash
    ollama pull gemma3:4b # Default LLM
    ollama pull nomic-embed-text # Default embedding model for RAG
    # Pull other models if specified in your configs
    ```
4.  **Import RAG Documents:**
    Run the import script at least once to populate the ChromaDB collection:
    ```bash
    python src/rag/importdocs.py
    ```

## Running the Assistant

Execute the main script:

```bash
python main.py [OPTIONS]
```

**Key Command-Line Options:**

- `--asr-preset <name>`: Use a specific preset from `conf_asr.json` (default: `default`).
- `--tts-preset <name>`: Use a specific preset from `conf_tts.json` (default: `default`).
- `--llm-preset <name>`: Use a specific preset from `conf_llm.json` (default: `default`).
- `--system-prompt "<prompt>"`: Override the LLM system prompt defined in the config.
- `--timeout <seconds>`: Speech detection timeout (default: 5).
- `--phrase-limit <seconds>`: Maximum duration for a single user phrase (default: 10).
- `--fixed-duration <seconds>`: Use fixed-duration recording instead of dynamic listening.
- `--asr-config <path>`, `--tts-config <path>`, `--llm-config <path>`: Specify alternative paths to configuration files.

## Configuration

Configuration is managed through JSON files located in `src/config/`:

- `conf_asr.json`: Settings for Automatic Speech Recognition (ASR), audio processing, and VAD parameters.
- `conf_tts.json`: Settings for Text-to-Speech (TTS).
- `conf_llm.json`: Settings for the Large Language Model (LLM), including RAG and generation parameters.

Each file can contain multiple "presets". You can select which preset to use via command-line arguments (e.g., `--asr-preset faster`). The `default` preset is used if none is specified.

Key settings to potentially adjust:

- `model_id` (in ASR, TTS, LLM configs): Change the specific models used.
- `device`, `compute_type` (in ASR/faster-whisper config): Optimize for your hardware (e.g., `cuda`, `float16`).
- `vad_energy_threshold`, `vad_confidence_threshold` (in ASR config `audio_validation`): Tune interrupt sensitivity.
- `temperature`, `top_p`, `top_k` (in LLM config): Adjust LLM generation creativity/determinism.
- `voice`, `speed`, `expressiveness` (in TTS config `kokoro`): Customize TTS output.
- `system_prompt` (in LLM config): Change the assistant's base personality/instructions.
