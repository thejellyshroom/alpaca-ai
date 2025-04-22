import traceback
import pyaudio
import wave
import numpy as np
import speech_recognition as sr
import threading
import queue
import time
import sounddevice as sd
import soundfile as sf
import io
import os
import math # Added for RMS calculation

from .audio_player import AudioPlayer
from .interrupt_detector import InterruptDetector

class AudioHandler:
    def __init__(self, config=None):
        self.config = config or {}
        self.audio_validation = self.config.get('audio_validation', {})
        self.recognizer_config = self.config.get('recognizer', {})
        self.pyaudio_instance = pyaudio.PyAudio()

        player_config = {
            'default_sample_rate': self.config.get('tts_sample_rate', 22050) # Example config key
        }
        self.player = AudioPlayer(self.pyaudio_instance, default_sample_rate=player_config['default_sample_rate'])

        detector_config = {
            'sample_rate': self.config.get('sample_rate', 44100),
            'channels': self.config.get('channels', 1),
            'chunk': self.config.get('chunk', 1024),
            'vad_energy_threshold': self.audio_validation.get('vad_energy_threshold', 300),
            'vad_activation_chunks': self.audio_validation.get('vad_activation_chunks', 3)
        }
        self.detector = InterruptDetector(self.pyaudio_instance, detector_config)

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = self.recognizer_config.get('pause_threshold', 1.5)
        self.recognizer.phrase_threshold = self.recognizer_config.get('phrase_threshold', 0.3)
        self.recognizer.non_speaking_duration = self.recognizer_config.get('non_speaking_duration', 1.0)
        self.recognizer.energy_threshold = self.recognizer_config.get('energy_threshold') # Let sr handle default if None
        self.recognizer.dynamic_energy_threshold = self.recognizer_config.get('dynamic_energy_threshold', True)

        # Parameters used directly by listen_for_speech
        self.max_retries = self.audio_validation.get('max_retries', 3)
        self.min_energy = self.audio_validation.get('min_energy', 100) # Provide a default min energy
        self.max_phrase_duration = self.audio_validation.get('max_phrase_duration', 300) # 5 minutes default

    # --- Property for Playback Status ---
    @property
    def is_playing(self):
        return self.player.is_playing

    # --- Core Listening Method ---
    def listen_for_speech(self, filename="prompt.wav", timeout=None, stop_playback=False):
        if stop_playback:
            try:
                self.stop_playback()
                self.player.wait_for_playback_complete(timeout=1.0)
            except Exception as e:
                print(f"Error stopping playback before listen: {e}")

        original_pause_threshold = self.recognizer.pause_threshold
        original_phrase_threshold = self.recognizer.phrase_threshold
        original_non_speaking_duration = self.recognizer.non_speaking_duration

        try:
            # Temporarily adjust settings for potentially better capture during listen
            self.recognizer.pause_threshold = self.recognizer_config.get('listen_pause_threshold', 1.5)
            self.recognizer.phrase_threshold = self.recognizer_config.get('listen_phrase_threshold', 0.3)
            self.recognizer.non_speaking_duration = self.recognizer_config.get('listen_non_speaking_duration', 1.0)

            retry_count = 0
            while retry_count <= self.max_retries:
                try:
                    # Use the VAD sample rate for the microphone to ensure compatibility
                    with sr.Microphone(sample_rate=self.detector.vad_sample_rate) as source:
                        duration = 1.0 if retry_count == 0 else 0.5
                        print(f"Adjusting for ambient noise ({duration}s)...")
                        self.recognizer.adjust_for_ambient_noise(source, duration=duration)

                        print(f"Listening with timeout={timeout if timeout else 5} seconds, phrase limit={self.max_phrase_duration}s...")
                        audio_data = self.recognizer.listen(
                            source,
                            timeout=timeout if timeout else 5,
                            phrase_time_limit=self.max_phrase_duration
                        )

                        current_energy_threshold = self.recognizer.energy_threshold
                        print(f"Listening finished. Energy Threshold during listen: {current_energy_threshold:.2f}")

                        if current_energy_threshold < self.min_energy: # Check against configured minimum required energy
                           print(f"Warning: Ambient noise level ({current_energy_threshold:.2f}) might be too low or recording failed. Energy lower than required min_energy ({self.min_energy}).")
                           if retry_count < self.max_retries:
                                print(f"Retrying listen due to low ambient energy (attempt {retry_count + 1}/{self.max_retries})...")
                                retry_count += 1
                                time.sleep(0.5) # Small delay before retry
                                continue
                           else:
                                return "low_energy"

                        # Save the audio file
                        wav_filename = filename if filename.endswith('.wav') else f"{filename}.wav"
                        filepath = os.path.abspath(wav_filename)
                        print(f"Saving audio to {filepath}...")
                        with open(filepath, "wb") as f:
                            f.write(audio_data.get_wav_data())
                        print(f"Audio saved successfully.")
                        return filepath

                except sr.WaitTimeoutError:
                     print("No speech detected within the timeout period.")
                     if retry_count < self.max_retries:
                         print(f"Retrying listen (attempt {retry_count + 1}/{self.max_retries})...")
                         retry_count += 1
                         continue
                     else:
                         return "TIMEOUT_ERROR"

                except Exception as e:
                    print(f"Unexpected error during listening attempt: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    if retry_count < self.max_retries:
                        print(f"Retrying listen due to error (attempt {retry_count + 1}/{self.max_retries})...")
                        retry_count += 1
                        time.sleep(1) # Longer delay after error
                        continue
                    return None # General error after retries

            print("Maximum retries exceeded. No valid speech detected.")
            return "TIMEOUT_ERROR" # Return timeout error after max retries

        except Exception as e:
            print(f"Error setting up microphone or during listen: {e}")
            traceback.print_exc()
            return None
        finally:
            # Restore original recognizer settings
            self.recognizer.pause_threshold = original_pause_threshold
            self.recognizer.phrase_threshold = original_phrase_threshold
            self.recognizer.non_speaking_duration = original_non_speaking_duration
            print("Listening session ended.")

    def play_audio(self, audio_data, sample_rate=None):
        """Delegate audio playback to the AudioPlayer."""
        self.player.play_audio(audio_data, sample_rate)

    def wait_for_playback_complete(self, timeout=None):
        """Delegate waiting for playback to the AudioPlayer."""
        return self.player.wait_for_playback_complete(timeout)

    def start_interrupt_listener(self, interrupt_event):
        """Delegate starting the interrupt listener to the InterruptDetector."""
        self.detector.start_interrupt_listener(interrupt_event)

    def stop_interrupt_listener(self):
        """Delegate stopping the interrupt listener to the InterruptDetector."""
        self.detector.stop_interrupt_listener()

    def stop_playback(self, force=False):
        """Stop both audio playback and the interrupt listener."""
        print("AudioHandler stopping playback and interrupt listener...")
        self.player.stop_playback(force)
        self.detector.stop_interrupt_listener() # Ensure detector is stopped too

    # --- Cleanup Method ---
    def __del__(self):
        """Cleanup resources for player, detector, and PyAudio."""
        try:
            if hasattr(self, 'player') and self.player:
                self.player.cleanup()
            if hasattr(self, 'detector') and self.detector:
                self.detector.cleanup()

            if hasattr(self, 'pyaudio_instance') and self.pyaudio_instance:
                time.sleep(0.2)
                self.pyaudio_instance.terminate()

        except Exception as e:
            print(f"Error during AudioHandler cleanup: {e}")
            traceback.print_exc()
        finally:
             print("AudioHandler cleanup finished.") 