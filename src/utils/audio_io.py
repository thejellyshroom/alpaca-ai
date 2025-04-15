import numpy as np
import soundfile as sf
import sounddevice as sd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional


def save_audio_file(
    audio_data: np.ndarray, output_dir: Path, sample_rate: int = 24000
) -> Path:
    """
    Save audio data to a WAV file with a timestamp in the filename.

    Args:
        audio_data (np.ndarray): The audio data to save. Can be a single array or a list of arrays.
        output_dir (Path): The directory to save the audio file in.
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 24000.

    Returns:
        Path: The path to the saved audio file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"output_{timestamp}.wav"

    if isinstance(audio_data, list):
        audio_data = np.concatenate(audio_data)

    sf.write(str(output_path), audio_data, sample_rate)
    return output_path


def play_audio(
    audio_data: np.ndarray, sample_rate: int = 24000
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Play audio data using sounddevice.

    Args:
        audio_data (np.ndarray): The audio data to play.
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 24000.

    Returns:
        Tuple[bool, Optional[np.ndarray]]: A tuple containing a boolean indicating if the playback was interrupted (always False here) and an optional numpy array representing the interrupted audio (always None here).
    """
    sd.play(audio_data, sample_rate)
    sd.wait()
    return False, None
