# Alpaca Voice Assistant

A locally-runnable AI voice and text assistant leveraging powerful open-source models for transcription, language understanding, and speech synthesis. Designed for responsiveness with streaming TTS capabilities and interruptible playback in voice mode.

## Features

- **Local First:** Runs entirely on your machine (requires Ollama).
- **Voice and Text Modes:** Interact via voice or text input/output.
- **Modular Architecture:** Core components (audio, STT, LLM, TTS, conversation, config) are separated for better maintainability.
- **Streaming TTS (Voice Mode):** Assistant starts speaking as soon as the first part of the response is generated.
- **Interruptible Playback (Voice Mode):** Uses Silero VAD and RMS energy detection to allow the user to interrupt the assistant mid-speech.
- **RAG Integration:** Utilizes Retrieval-Augmented Generation via `minirag` (which internally uses Ollama embeddings) to answer questions based on imported documents in the `DATA_PATH`.
- **Configurable:** Easily configure models, presets, VAD sensitivity, and other parameters via JSON files and `.env`.
- **Session Summarization:** Automatically saves a summary of the conversation upon exit.
- **Models Used (Defaults):**
  - **ASR:** `faster-whisper` (`Systran/faster-whisper-small`)
  - **LLM:** `ollama` (configurable via `.env`, e.g., `gemma3:4b`)
  - **TTS:** Kokoro TTS
  - **Embedding (for RAG):** Configurable via `.env` (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
  - **VAD:** `silero-vad`

## Architecture

- `main.py`: Entry point, handles initialization, argument parsing (mode selection), RAG indexing, configuration loading, and starts the main interaction loop. Catches `KeyboardInterrupt` for graceful shutdown and summarization.
- `src/core/`: Contains the core orchestration classes:
  - `alpaca.py`: Main container class (`Alpaca`).
  - `alpaca_interaction.py`: Handles the logic for a single interaction turn (`AlpacaInteraction`). Contains methods for voice and text interaction.
  - `voice_loop.py`: Contains the synchronous main loop function (`run_voice_interaction_loop`) for voice mode.
  - `text_loop.py`: Contains the synchronous main loop function (`run_text_interaction_loop`) for text mode.
- `src/utils/`: Contains utility classes and functions:
  - `component_manager.py`: Manages the lifecycle (loading, access, cleanup) of handlers.
  - `conversation_manager.py`: Manages the conversation history.
  - `config_loader.py`: Handles loading JSON configurations based on presets.
  - `summarizer.py`: Handles conversation summarization logic.
- `src/components/`: Contains specialized handlers for specific tasks:
  - `audio_handler.py`: Coordinates audio input (listening) and output.
  - `audio_player.py`: Handles audio playback queue.
  - `interrupt_detector.py`: Handles VAD and RMS energy detection for interruptions (voice mode).
  - `output_handler.py`: Handles TTS synthesis and speaking logic, including streaming and interruption handling for voice mode.
  - `stt_handler.py`: Interface for the chosen speech-to-text model (e.g., `Transcriber`).
  - `llm_handler.py`: Interface for the chosen large language model (e.g., Ollama) and RAG querying.
  - `tts_handler.py`: Interface for the chosen text-to-speech model.
- `src/rag/`: Contains scripts and modules related to RAG:
  - `indexer.py`: Script run at startup to index documents from `DATA_PATH` using `minirag`.
  - `minirag/`: Submodule containing the MiniRAG library for embedding, indexing, and querying.
- `src/config/`: Contains JSON configuration files (`conf_asr.json`, `conf_tts.json`, `conf_llm.json`).

## Setup & Installation

**Prerequisites:**

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running.
- Microphone connected and configured (for voice mode).
- Speakers/Headphones (for voice mode).

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
    - Install required packages (refer to `requirements.txt`):
      ```bash
      pip install -r requirements.txt
      # If requirements.txt is missing, manually install key packages:
      # pip install -U torch torchaudio torchvision # Ensure compatibility!
      # pip install -U ollama faster-whisper SpeechRecognition pyaudio sounddevice soundfile numpy
      # pip install -U silero-vad onnxruntime # For VAD
      # pip install -U nltk rouge-score transformers # For RAG/utils
      # pip install -U python-dotenv json_repair # For config/utils
      # Add any other specific dependencies
      ```
      _Note: Verify PyTorch version compatibility._
3.  **Set up Environment Variables:**
    - Copy the example environment file (if provided, e.g., `.env.example`) to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Edit `.env` to configure essential paths and models:
      - `DATA_PATH`: Path to the directory containing documents for RAG (e.g., `./data/dataset`). Create this directory if it doesn't exist.
      - `WORKING_DIR`: Path for MiniRAG cache and index files (e.g., `./minirag_cache`).
      - `EMBEDDING_MODEL`: Model used for RAG embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2` or an Ollama model like `nomic-embed-text`).
      - `EXTRACTION_LLM_MODEL`: Ollama model used by MiniRAG for internal processing (e.g., `gemma3:4b`).
      - `QUERY_LLM_MODEL`: Ollama model used by MiniRAG for generating answers from retrieved context (e.g., `gemma3:4b`).
      - `ENABLE_RAG`: Set to `true` to enable RAG, `false` otherwise.
      - `DEFAULT_MODE`: Set to `voice` or `text`.
4.  **Download Ollama Models:**
    Ensure the models specified in your `.env` and `conf_llm.json` are pulled:
    ```bash
    ollama pull gemma3:4b # Example LLM
    ollama pull nomic-embed-text # Example embedding model
    ```
5.  **Add RAG Documents:**
    Place the text files (`.txt`) you want the assistant to reference into the directory specified by `DATA_PATH` in your `.env` file. The `indexer.py` script will automatically process these when `main.py` starts.

## Running the Assistant

Execute the main script:

```bash
python main.py [OPTIONS]
```

**Key Command-Line Options:**

- `--mode <voice|text>`: Overrides the `DEFAULT_MODE` from `.env`. Run in voice or text mode.
- `--asr-preset <name>`: Use a specific preset from `conf_asr.json` (default: `default`).
- `--tts-preset <name>`: Use a specific preset from `conf_tts.json` (default: `default`).
- `--llm-preset <name>`: Use a specific preset from `conf_llm.json` (default: `default`).
- `--timeout <seconds>`: Speech detection timeout (voice mode) (default: 5).
- `--phrase-limit <seconds>`: Maximum duration for a single user phrase (voice mode) (default: 10).
- `--asr-config <path>`, `--tts-config <path>`, `--llm-config <path>`: Specify alternative paths to configuration files.

## Configuration

Configuration is primarily managed through:

- **`.env` file:** For essential paths, RAG settings, and default models.
- **JSON files in `src/config/`:** For detailed component settings and presets.
  - `conf_asr.json`: Settings for Automatic Speech Recognition (ASR), audio processing, and VAD parameters.
  - `conf_tts.json`: Settings for Text-to-Speech (TTS).
  - `conf_llm.json`: Settings for the Large Language Model (LLM) generation parameters (e.g., temperature).

You can select JSON presets via command-line arguments (e.g., `--asr-preset faster`). The `default` preset is used if none is specified.

Key settings to potentially adjust:

- **`.env**:\*\* `DATA_PATH`, `WORKING_DIR`, `EMBEDDING_MODEL`, `EXTRACTION_LLM_MODEL`, `QUERY_LLM_MODEL`, `ENABLE_RAG`, `DEFAULT_MODE`.
- **JSON configs:**
  - `model_id` (in ASR, TTS): Change the specific models used.
  - `device`, `compute_type` (in ASR/faster-whisper config): Optimize for your hardware (e.g., `cuda`, `float16`).
  - `vad_energy_threshold`, `vad_confidence_threshold` (in ASR config `audio_validation`): Tune interrupt sensitivity (voice mode).
  - `temperature`, `top_p`, `top_k` (in LLM config): Adjust LLM generation creativity/determinism.
  - `voice`, `speed` (in TTS configs): Customize TTS output (voice mode).
  - `system_prompt` (in LLM config `default`): Change the assistant's base personality/instructions (can be overridden by `.env`).
