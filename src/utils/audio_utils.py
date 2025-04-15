import time
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import numpy as np
from .audio_io import save_audio_file, play_audio
from .audio_queue import AudioGenerationQueue


def generate_and_play_sentences(
    sentences: List[str],
    generator,
    speed: float = 1.0,
    play_function: Callable = play_audio,
    check_interrupt: Optional[Callable] = None,
    output_dir: Optional[Path] = None,
    sample_rate: Optional[int] = None,
) -> Tuple[bool, Optional[np.ndarray], List[Path]]:
    """
    Generates and plays audio for each sentence with optional interruption checking.

    Args:
        sentences (List[str]): A list of sentences to generate audio for.
        generator: The audio generator object.
        speed (float, optional): The speed of audio generation. Defaults to 1.0.
        play_function (Callable, optional): The function to use for playing audio. Defaults to play_audio.
        check_interrupt (Callable, optional): An optional function to check for interruptions. Defaults to None.
        output_dir (Path, optional): The directory to save generated audio files. Defaults to None.
        sample_rate (int, optional): The sample rate of the audio. Defaults to None.

    Returns:
        Tuple[bool, Optional[np.ndarray], List[Path]]: A tuple containing:
            - A boolean indicating if the process was interrupted.
            - Optional audio data if the process was interrupted.
            - A list of paths to the generated audio files.
    """
    audio_queue = AudioGenerationQueue(generator, speed, output_dir)
    audio_queue.start()
    audio_queue.add_sentences(sentences)

    audio_files = []
    was_interrupted = False
    interrupt_audio = None

    try:
        while True:
            if check_interrupt:
                interrupted, audio_data = check_interrupt()
                if interrupted:
                    was_interrupted = True
                    interrupt_audio = audio_data
                    break

            audio_data, output_path = audio_queue.get_next_audio()

            if audio_data is not None:
                if output_path:
                    audio_files.append(output_path)

                if play_function:
                    try:
                        was_interrupted, interrupt_data = (
                            play_function(audio_data, sample_rate)
                            if sample_rate
                            else play_function(audio_data)
                        )
                        if was_interrupted:
                            interrupt_audio = interrupt_data
                            break
                    except Exception as e:
                        print(f"Error playing audio: {str(e)}")
                        continue

            if audio_queue.sentence_queue.empty() and audio_queue.audio_queue.empty():
                break

            time.sleep(0.01)

    except Exception as e:
        print(f"Error in generate_and_play_sentences: {str(e)}")
    finally:
        audio_queue.stop()

    return was_interrupted, interrupt_audio, audio_files
