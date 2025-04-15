from queue import Queue
import threading
import time
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from .audio_io import save_audio_file

logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("speechbrain.utils.quirks").setLevel(logging.ERROR)
logging.basicConfig(format="%(message)s", level=logging.INFO)


class AudioGenerationQueue:
    """
    A queue system for managing asynchronous audio generation from text input.

    This class implements a threaded queue system that handles text-to-audio generation
    in a background thread. It provides functionality for adding sentences to be processed,
    retrieving generated audio, and monitoring the generation process.

    Attributes:
        generator: Audio generator instance used for text-to-speech conversion
        speed (float): Speed multiplier for audio generation
        output_dir (Path): Directory where generated audio files are saved
        sentences_processed (int): Count of processed sentences
        audio_generated (int): Count of successfully generated audio files
        failed_sentences (list): List of tuples containing failed sentences and error messages
    """

    def __init__(
        self, generator, speed: float = 1.0, output_dir: Optional[Path] = None
    ):
        """
        Initialize the audio generation queue system.

        Args:
            generator: Audio generator instance for text-to-speech conversion
            speed: Speed multiplier for audio generation (default: 1.0)
            output_dir: Directory path for saving generated audio files (default: "generated_audio")
        """
        self.generator = generator
        self.speed = speed
        self.lock = threading.Lock()
        self.output_dir = output_dir or Path("generated_audio")
        self.output_dir.mkdir(exist_ok=True)
        self.sentence_queue = Queue()
        self.audio_queue = Queue()
        self.is_running = False
        self.generation_thread = None
        self.sentences_processed = 0
        self.audio_generated = 0
        self.failed_sentences = []

    def start(self):
        """
        Start the audio generation thread if not already running.
        The thread will process sentences from the queue until stopped.
        """
        if not self.is_running:
            self.is_running = True
            self.generation_thread = threading.Thread(target=self._generation_worker)
            self.generation_thread.daemon = True
            self.generation_thread.start()

    def stop(self):
        """
        Stop the audio generation thread gracefully.
        Waits for the current queue to be processed before stopping.
        Outputs final processing statistics.
        """
        if self.generation_thread:
            while not self.sentence_queue.empty():
                time.sleep(0.1)

            time.sleep(0.5)

            self.is_running = False
            self.generation_thread.join()
            self.generation_thread = None

            logging.info(
                f"\nAudio Generation Complete - Processed: {self.sentences_processed}, Generated: {self.audio_generated}, Failed: {len(self.failed_sentences)}"
            )

    def add_sentences(self, sentences: List[str]):
        """
        Add a list of sentences to the generation queue.

        Args:
            sentences: List of text strings to be converted to audio
        """
        added_count = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                self.sentence_queue.put(sentence)
                added_count += 1

        if not self.is_running:
            self.start()

    def get_next_audio(self) -> Tuple[Optional[np.ndarray], Optional[Path]]:
        """
        Retrieve the next generated audio segment from the queue.

        Returns:
            Tuple containing:
                - numpy array of audio data (or None if queue is empty)
                - Path object for the saved audio file (or None if queue is empty)
        """
        try:
            audio_data, output_path = self.audio_queue.get_nowait()
            return audio_data, output_path
        except:
            return None, None

    def clear_queues(self):
        """
        Clear both sentence and audio queues, removing all pending items.
        Returns immediately without waiting for queue processing.
        """
        sentences_cleared = 0
        audio_cleared = 0

        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
                sentences_cleared += 1
            except:
                pass

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                audio_cleared += 1
            except:
                pass

    def _generation_worker(self):
        """
        Internal worker method that runs in a separate thread.
        Continuously processes sentences from the queue, generating audio
        and handling any errors that occur during generation.
        """
        while self.is_running or not self.sentence_queue.empty():
            try:
                try:
                    sentence = self.sentence_queue.get_nowait()
                    self.sentences_processed += 1
                except:
                    if not self.is_running and self.sentence_queue.empty():
                        break
                    time.sleep(0.01)
                    continue

                try:
                    audio_data, phonemes = self.generator.generate(
                        sentence, speed=self.speed
                    )

                    if audio_data is None or len(audio_data) == 0:
                        raise ValueError("Generated audio data is empty")

                    output_path = save_audio_file(audio_data, self.output_dir)
                    self.audio_generated += 1

                    self.audio_queue.put((audio_data, output_path))

                except Exception as e:
                    error_msg = str(e)
                    self.failed_sentences.append((sentence, error_msg))
                    continue

            except Exception as e:
                if not self.is_running and self.sentence_queue.empty():
                    break
                time.sleep(0.1)
