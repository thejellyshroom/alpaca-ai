import pyaudio
import numpy as np
import torch
from torch.nn.functional import pad
import time
from queue import Queue
import sounddevice as sd
from .config import settings
import traceback
from typing import Union

CHUNK = settings.CHUNK
FORMAT = pyaudio.paFloat32
CHANNELS = settings.CHANNELS
RATE = settings.RATE
SILENCE_THRESHOLD = settings.SILENCE_THRESHOLD
SPEECH_CHECK_THRESHOLD = settings.SPEECH_CHECK_THRESHOLD
MAX_SILENCE_DURATION = settings.MAX_SILENCE_DURATION


def init_vad_pipeline(hf_token):
    """Initializes the Voice Activity Detection pipeline.

    Args:
        hf_token (str): Hugging Face API token.

    Returns:
        pyannote.audio.pipelines.VoiceActivityDetection: VAD pipeline.
    """
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    model = Model.from_pretrained(settings.VAD_MODEL, use_auth_token=hf_token)

    pipeline = VoiceActivityDetection(segmentation=model)

    HYPER_PARAMETERS = {
        "min_duration_on": settings.VAD_MIN_DURATION_ON,
        "min_duration_off": settings.VAD_MIN_DURATION_OFF,
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    return pipeline


def detect_speech_segments(pipeline, audio_data, sample_rate=None):
    """Detects speech segments in audio using pyannote VAD.

    Args:
        pipeline (pyannote.audio.pipelines.VoiceActivityDetection): VAD pipeline.
        audio_data (np.ndarray or torch.Tensor): Audio data.
        sample_rate (int, optional): Sample rate of the audio. Defaults to settings.RATE.

    Returns:
        torch.Tensor or None: Concatenated speech segments as a torch tensor, or None if no speech is detected.
    """
    if sample_rate is None:
        sample_rate = settings.RATE

    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(1, -1)

    if not isinstance(audio_data, torch.Tensor):
        audio_data = torch.from_numpy(audio_data)

    if audio_data.shape[1] < sample_rate:
        padding_size = sample_rate - audio_data.shape[1]
        audio_data = pad(audio_data, (0, padding_size))

    vad = pipeline({"waveform": audio_data, "sample_rate": sample_rate})

    speech_segments = []
    for speech in vad.get_timeline().support():
        start_sample = int(speech.start * sample_rate)
        end_sample = int(speech.end * sample_rate)
        if start_sample < audio_data.shape[1]:
            end_sample = min(end_sample, audio_data.shape[1])
            segment = audio_data[0, start_sample:end_sample]
            speech_segments.append(segment)

    if speech_segments:
        return torch.cat(speech_segments)
    return None


def record_audio(duration=None):
    """Records audio for a specified duration.

    Args:
        duration (int, optional): Recording duration in seconds. Defaults to settings.RECORD_DURATION.

    Returns:
        np.ndarray: Recorded audio data as a numpy array.
    """
    if duration is None:
        duration = settings.RECORD_DURATION

    p = pyaudio.PyAudio()

    stream = p.open(
        format=settings.FORMAT,
        channels=settings.CHANNELS,
        rate=settings.RATE,
        input=True,
        frames_per_buffer=settings.CHUNK,
    )

    print("\nRecording...")
    frames = []

    for i in range(0, int(settings.RATE / settings.CHUNK * duration)):
        data = stream.read(settings.CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.concatenate(frames, axis=0)
    return audio_data


def record_continuous_audio():
    """Continuously monitors audio and detects speech segments.

    Returns:
        np.ndarray or None: Recorded audio data as a numpy array, or None if no speech is detected.
    """
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("\nListening... (Press Ctrl+C to stop)")
    frames = []
    buffer_frames = []
    buffer_size = int(RATE * 0.5 / CHUNK)
    silence_frames = 0
    max_silence_frames = int(RATE / CHUNK * 1)
    recording = False

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            buffer_frames.append(audio_chunk)
            if len(buffer_frames) > buffer_size:
                buffer_frames.pop(0)

            audio_level = np.abs(np.concatenate(buffer_frames)).mean()

            if audio_level > SILENCE_THRESHOLD:
                if not recording:
                    print("\nPotential speech detected...")
                    recording = True
                    frames.extend(buffer_frames)
                frames.append(audio_chunk)
                silence_frames = 0
            elif recording:
                frames.append(audio_chunk)
                silence_frames += 1

                if silence_frames >= max_silence_frames:
                    print("Processing speech segment...")
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if frames:
        return np.concatenate(frames)
    return None


def check_for_speech(timeout=0.1):
    """Checks if speech is detected in a non-blocking way.

    Args:
        timeout (float, optional): Duration to check for speech in seconds. Defaults to 0.1.

    Returns:
        tuple: A tuple containing a boolean indicating if speech was detected and the audio data as a numpy array, or (False, None) if no speech is detected.
    """
    p = pyaudio.PyAudio()

    frames = []
    is_speech = False

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        for _ in range(int(RATE * timeout / CHUNK)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)

            audio_level = np.abs(audio_chunk).mean()
            if audio_level > SPEECH_CHECK_THRESHOLD:
                is_speech = True
                break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if is_speech and frames:
        return True, np.concatenate(frames)
    return False, None


def play_audio_with_interrupt(audio_data, sample_rate=24000):
    """Plays audio while monitoring for speech interruption.

    Args:
        audio_data (np.ndarray): Audio data to play.
        sample_rate (int, optional): Sample rate for playback. Defaults to 24000.

    Returns:
        tuple: A tuple containing a boolean indicating if playback was interrupted and None, or (False, None) if playback completes without interruption.
    """
    interrupt_queue = Queue()

    def input_callback(indata, frames, time, status):
        """Callback for monitoring input audio."""
        if status:
            print(f"Input status: {status}")
            return

        audio_level = np.abs(indata[:, 0]).mean()
        if audio_level > settings.INTERRUPTION_THRESHOLD:
            interrupt_queue.put(True)

    def output_callback(outdata, frames, time, status):
        """Callback for output audio."""
        if status:
            print(f"Output status: {status}")
            return

        if not interrupt_queue.empty():
            raise sd.CallbackStop()

        remaining = len(audio_data) - output_callback.position
        if remaining == 0:
            raise sd.CallbackStop()
        valid_frames = min(remaining, frames)
        outdata[:valid_frames, 0] = audio_data[
            output_callback.position : output_callback.position + valid_frames
        ]
        if valid_frames < frames:
            outdata[valid_frames:] = 0
        output_callback.position += valid_frames

    output_callback.position = 0

    try:
        with sd.InputStream(
            channels=1, callback=input_callback, samplerate=settings.RATE
        ):
            with sd.OutputStream(
                channels=1, callback=output_callback, samplerate=sample_rate
            ):
                while output_callback.position < len(audio_data):
                    sd.sleep(100)
                    if not interrupt_queue.empty():
                        return True, None
        return False, None
    except sd.CallbackStop:
        return True, None
    except Exception as e:
        print(f"Error during playback: {str(e)}")
        return False, None


def record_with_timeout(timeout: float, threshold: float) -> Union[np.ndarray, None]:
    """Records audio for a specific duration, returning audio only if sound exceeds threshold.

    Args:
        timeout (float): Recording duration in seconds.
        threshold (float): Audio level threshold to detect sound.

    Returns:
        np.ndarray | None: Recorded audio data if sound was detected, otherwise None.
    """
    p = pyaudio.PyAudio()
    stream = None
    frames = []
    sound_detected = False

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        num_chunks = int(RATE * timeout / CHUNK)
        # print(f"Listening for {timeout} seconds ({num_chunks} chunks)...") # Optional debug print

        for _ in range(num_chunks):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                frames.append(audio_chunk)

                # Check if audio level in this chunk exceeds the threshold
                if np.abs(audio_chunk).mean() > threshold:
                    sound_detected = True
                    # print(f"Sound detected! Level: {np.abs(audio_chunk).mean():.4f}") # Optional debug print

            except IOError as e:
                # Handle input overflow gracefully if it occurs
                if e.errno == pyaudio.paInputOverflowed:
                    print("Warning: Input overflowed. Skipping chunk.")
                else:
                    raise # Re-raise other IOErrors

        # print("Finished listening.") # Optional debug print

    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()

    if sound_detected and frames:
        # print("Returning recorded audio.") # Optional debug print
        return np.concatenate(frames)
    else:
        # print("No significant sound detected within timeout.") # Optional debug print
        return None


def transcribe_audio(model, audio_data):
    """Transcribes audio data using a Faster Whisper model.

    Args:
        model (faster_whisper.WhisperModel): The loaded Faster Whisper model.
        audio_data (np.ndarray | torch.Tensor | list | tuple): The audio data.

    Returns:
        str: The transcribed text.
    """
    try:
        # Convert torch.Tensor to numpy array first
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy()

        # Ensure audio_data is float32 numpy array (faster-whisper expects this)
        if not isinstance(audio_data, np.ndarray):
            # If it's a list or tuple of numbers, convert it
            if isinstance(audio_data, (list, tuple)) and all(isinstance(x, (int, float)) for x in audio_data):
                audio_data = np.array(audio_data, dtype=np.float32)
            else:
                raise TypeError("audio_data must be a numpy array or a list/tuple of numbers")

        if audio_data.dtype != np.float32:
            # Attempt conversion if possible, otherwise raise error
            try:
                audio_data = audio_data.astype(np.float32)
            except Exception as e:
                raise TypeError(f"Could not convert audio_data to float32: {e}")

        # Check if audio data is effectively silent or too short
        if np.max(np.abs(audio_data)) < 0.001 or len(audio_data) < 100: # Adjust threshold/length as needed
            print("Audio data is silent or too short, skipping transcription.")
            return ""

        # Transcribe using faster-whisper
        # You might want to add parameters like language detection if needed:
        # segments, info = model.transcribe(audio_data, beam_size=5, language="en")
        segments, info = model.transcribe(audio_data, beam_size=5)

        # Concatenate segments into a single string
        transcription = "".join(segment.text for segment in segments)

        return transcription.strip()

    except Exception as e:
        print(f"Error during transcription: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return ""
