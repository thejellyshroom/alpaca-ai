import numpy as np
import sounddevice as sd
import time


def play_audio(audio_data: np.ndarray, sample_rate: int = 24000):
    """
    Play audio directly using sounddevice.

    Args:
        audio_data (np.ndarray): The audio data to play.
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 24000.
    """
    try:
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {str(e)}")


def stream_audio_chunks(
    audio_chunks: list, sample_rate: int = 24000, pause_duration: float = 0.2
):
    """
    Stream audio chunks one after another with a small pause between them.

    Args:
        audio_chunks (list): A list of audio chunks to play.
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 24000.
        pause_duration (float, optional): The duration of the pause between chunks in seconds. Defaults to 0.2.
    """
    try:
        for chunk in audio_chunks:
            if len(chunk) == 0:
                continue
            sd.play(chunk, sample_rate)
            sd.wait()
            time.sleep(pause_duration)
    except Exception as e:
        print(f"Error streaming audio chunks: {str(e)}")
    finally:
        sd.stop()
