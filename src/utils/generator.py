import torch
import numpy as np
from pathlib import Path
from .voice import split_into_sentences
from kokoro import KPipeline
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)

class VoiceGenerator:
    """
    A class to manage voice generation using a pre-trained model.
    """

    def __init__(self, voices_dir):
        """
        Initializes the VoiceGenerator with voice directory.

        Args:
            voices_dir (Path): Path to the directory containing voice pack files.
        """
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.pipeline = None
        self.voice_name = None
        self.speed = 1.0
        self.voices_dir = voices_dir
        self._initialized = False

    def initialize(self, voice_name, speed):
        """
        Initializes the Kokoro KPipeline.

        Args:
            voice_name (str): The name of the voice pack (e.g., 'af_heart').
            speed (float): The base playback speed.

        Returns:
            str: A message indicating the voice has been loaded.

        Raises:
            Exception: If there is an error during pipeline initialization.
        """
        try:
            logger.info(f"Initializing Kokoro TTS with voice: {voice_name}")
            lang_code = voice_name[0]
            self.pipeline = KPipeline(lang_code=lang_code)
            self.voice_name = voice_name
            self.speed = speed
            self._initialized = True
            logger.info(f"Successfully initialized Kokoro pipeline with voice: {voice_name}, speed: {speed}")
            return f"Loaded voice: {voice_name}"

        except Exception as e:
            logger.error(f"Failed to initialize Kokoro KPipeline: {e}")
            raise RuntimeError(f"Failed to initialize Kokoro KPipeline. Error: {e}")

    def list_available_voices(self):
        """
        Lists all available voice packs in the voices directory.
        (Note: This might need updating if voices aren't stored as .pt files anymore)

        Returns:
            list: A list of voice pack names (without the .pt extension).
        """
        return [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "e_asif", "e_cassie", "e_emma", "e_jack", "e_jeremy", "e_josh", "e_lucy", "e_maria"
        ]

    def is_initialized(self):
        """
        Checks if the generator is properly initialized.

        Returns:
            bool: True if the pipeline is initialized, False otherwise.
        """
        return self._initialized and self.pipeline is not None

    def generate(
        self,
        text,
        speed=None,
        pause_duration=200,
        short_text_limit=200,
        return_chunks=False,
    ):
        """
        Generates speech from the given text using Kokoro KPipeline.

        Handles both short and long-form text by splitting long text into sentences.

        Args:
            text (str): The text to generate speech from.
            speed (float, optional): Override the default speed for this generation. Defaults to None (use initialized speed).
            pause_duration (int, optional): Approximate pause duration between sentences (ms). Kokoro has internal sentence pausing.
            short_text_limit (int, optional): Character limit for direct processing. Defaults to 200.
            return_chunks (bool, optional): If True, returns a list of audio chunks instead of concatenated audio. Defaults to False.

        Returns:
            tuple: A tuple containing the generated audio (numpy array or list of numpy arrays) and an empty list (phonemes not provided).

        Raises:
            RuntimeError: If the pipeline is not initialized.
            ValueError: If there is an error during audio generation.
        """
        if not self.is_initialized():
            raise RuntimeError("Kokoro pipeline not initialized. Call initialize() first.")

        current_speed = speed if speed is not None else self.speed
        current_voice = self.voice_name

        text = text.strip()
        if not text:
            return (np.zeros(0, dtype=np.float32), []) if not return_chunks else ([], [])

        logger.debug(f"Generating audio for: '{text[:50]}...' with speed {current_speed} and voice {current_voice}")

        try:
            audio_segments = []
            generator = self.pipeline(
                text,
                voice=current_voice,
                speed=current_speed
            )

            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None:
                    if hasattr(audio, 'numpy'):
                        audio_np = audio.numpy().astype(np.float32)
                    elif isinstance(audio, np.ndarray):
                        audio_np = audio.astype(np.float32)
                    else:
                         logger.warning(f"Unexpected audio type from Kokoro: {type(audio)}. Skipping segment.")
                         continue

                    if audio_np.size > 0:
                        audio_segments.append(audio_np)
                    else:
                        logger.warning(f"Empty audio segment generated for: '{gs}'")
                else:
                    logger.warning(f"None audio segment generated for: '{gs}'")

            if not audio_segments:
                logger.error(f"No valid audio segments generated for text: '{text[:50]}...'")
                return (np.zeros(0, dtype=np.float32), []) if not return_chunks else ([], [])

            if return_chunks:
                return audio_segments, []
            else:
                combined_audio = np.concatenate(audio_segments)
                logger.debug(f"Generated combined audio of length {len(combined_audio)}")
                return combined_audio, []

        except Exception as e:
            logger.error(f"Error during Kokoro audio generation: {e}", exc_info=True)
            raise ValueError(f"Error in Kokoro audio generation: {str(e)}")
