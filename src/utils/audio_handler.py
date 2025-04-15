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

#========TODO: add energy check for silent audio and for shouting. 30 for whispers, 100 for normal speech, 200 for shouting
# shall be added to the config file and implemented in this way
class AudioHandler:
    def __init__(self, sample_rate=44100, channels=1, chunk=1024, config=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.pyaudio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        
        # Audio validation parameters
        self.config = config or {}
        self.audio_validation = self.config.get('audio_validation', {})
        self.min_duration = self.audio_validation.get('min_duration')
        self.min_file_size = self.audio_validation.get('min_file_size')
        self.max_retries = self.audio_validation.get('max_retries')
        self.timeout = self.audio_validation.get('timeout')
        self.min_energy = self.audio_validation.get('min_energy')
        self.max_phrase_duration = self.audio_validation.get('max_phrase_duration')  # Default 5 minutes

        # Recognizer parameters
        self.recognizer_config = self.config.get('recognizer', {})
        self.recognizer.pause_threshold = self.recognizer_config.get('pause_threshold')
        self.recognizer.phrase_threshold = self.recognizer_config.get('phrase_threshold')
        self.recognizer.non_speaking_duration = self.recognizer_config.get('non_speaking_duration')
        self.recognizer.energy_threshold = self.recognizer_config.get('energy_threshold')
        self.recognizer.dynamic_energy_threshold = self.recognizer_config.get('dynamic_energy_threshold')
        
        # Audio playback queue and thread
        self.audio_queue = queue.Queue(maxsize=100)
        self.playback_thread = None
        self.is_playing = False
        self.should_stop_playback = threading.Event()
        
        # Track total audio duration for dynamic timeout calculation
        self.total_audio_duration = 0.0
        self.last_audio_timestamp = 0.0
        
    def listen_for_speech(self, filename="prompt.wav", timeout=None, stop_playback=False):
        # Stop any ongoing playback if requested
        if stop_playback:
            try:
                self.stop_playback()
                # Wait for playback to complete, using a minimal timeout since we're stopping it anyway
                self.wait_for_playback_complete(timeout=2.0)
            except Exception as e:
                print(f"Error stopping playback: {e}")
        
        print("Listening for speech...")
        
        # Set parameters for better speech detection
        original_pause_threshold = self.recognizer.pause_threshold
        original_phrase_threshold = self.recognizer.phrase_threshold
        original_non_speaking_duration = self.recognizer.non_speaking_duration
        
        try:
            # Use balanced settings that won't cut off too early or wait too long
            self.recognizer.pause_threshold = 1.5      # Wait 1.5 seconds of silence before ending (balanced)
            self.recognizer.phrase_threshold = 0.3     # Detect speech relatively quickly
            self.recognizer.non_speaking_duration = 1.0  # Keep some silence but not too much
            
            retry_count = 0
            
            while retry_count <= self.max_retries:
                try:
                    with sr.Microphone() as source:
                        # Adjust for ambient noise with a longer duration for first attempt
                        duration = 1.0 if retry_count == 0 else 0.5
                        self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                        
                        # Listen for speech with proper timeout
                        print(f"Listening with timeout={timeout if timeout else 5} seconds...")
                        audio_data = self.recognizer.listen(
                            source, 
                            timeout=timeout if timeout else 5,  # Default timeout to prevent hanging
                            phrase_time_limit=self.max_phrase_duration  # Use configurable max duration
                        )
                        
                        # Before saving the file, check if we actually got meaningful audio
                        audio_duration = len(audio_data.frame_data) / (2 * 16000)
                        audio_energy = self.recognizer.energy_threshold
                        
                        print(f"Listening finished: Duration={audio_duration:.2f}s, Energy={audio_energy:.2f}")
                        
                        # ====== ENERGY CHECK ====== can be used for future checks of whether whispering or shouting
                        if audio_energy < self.min_energy: 
                            print(f"Warning: Very low energy audio detected (Energy={audio_energy:.2f}.")
                            # This is likely silence or background noise, treat as timeout
                            if retry_count < self.max_retries:
                                print(f"Retrying listen due to low energy (attempt {retry_count+1}/{self.max_retries})...")
                                retry_count += 1
                                continue
                            else:
                                return "low_energy"  # Treat as timeout error
                        
                        # Only save the file if we've passed all validation checks
                        wav_filename = filename if filename.endswith('.wav') else f"{filename}.wav"
                        with open(wav_filename, "wb") as f:
                            f.write(audio_data.get_wav_data())
                        
                        print(f"Audio saved as {wav_filename}")
                        return wav_filename
    
                except Exception as e:
                    print(f"Unexpected error during listening: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    if retry_count < self.max_retries:
                        print(f"Retrying listen due to error (attempt {retry_count+1}/{self.max_retries})...")
                        retry_count += 1
                        continue
                    return None
                    
            print("Maximum retries exceeded. No valid speech detected.")
            return None
            
        except Exception as e:
            print(f"Error during listening: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Restore original parameters
            self.recognizer.pause_threshold = original_pause_threshold
            self.recognizer.phrase_threshold = original_phrase_threshold
            self.recognizer.non_speaking_duration = original_non_speaking_duration

    def play_audio(self, audio_data, sample_rate=22050):
        """Play audio data through speakers.
        
        Args:
            audio_data (numpy.ndarray or torch.Tensor): Audio data as numpy array or PyTorch tensor
            sample_rate (int): Sample rate of the audio data
        """
        # Convert PyTorch tensor to numpy array if needed
        if hasattr(audio_data, 'detach') and hasattr(audio_data, 'cpu') and hasattr(audio_data, 'numpy'):
            # This is likely a PyTorch tensor
            audio_data = audio_data.detach().cpu().numpy()
        
        # Ensure audio data is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Calculate audio duration in seconds
        audio_duration = len(audio_data) / sample_rate
        
        # Update total audio duration for timeout calculations
        self.total_audio_duration += audio_duration
        self.last_audio_timestamp = time.time()
        
        # # Log audio duration for debugging
        # print(f"Adding audio segment: {audio_duration:.2f}s, total buffered: {self.total_audio_duration:.2f}s")
        
        # Start the playback thread if it's not already running
        self.start_playback_thread(sample_rate)
        
        # Add audio to the queue
        self.audio_queue.put((audio_data, sample_rate))
        self.is_playing = True
        
    def wait_for_playback_complete(self, timeout=None):
        """Wait until all audio playback has completed.
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds. If None, calculated dynamically.
            
        Returns:
            bool: True if playback completed, False if timed out
        """
        if not self.is_playing and self.audio_queue.empty():
            # Nothing is playing, reset tracking variables
            self.total_audio_duration = 0.0
            return True
        
        # Calculate dynamic timeout if not provided
        if timeout is None:
            # Base timeout on total audio duration plus buffer
            # Formula: audio_duration * 1.5 + 2.0 seconds (min 5 seconds, max 60 seconds)
            timeout = min(max(self.total_audio_duration * 1.5 + 2.0, 5.0), 60.0)
            
        # Log the calculated timeout    
        print(f"Waiting for audio playback to complete (timeout: {timeout:.1f}s, audio duration: {self.total_audio_duration:.1f}s)...")
        
        start_time = time.time()
        
        # More aggressive approach to waiting for playback to complete:
        # 1. First wait for the queue to be empty
        while not self.audio_queue.empty() and time.time() - start_time < timeout:
            time.sleep(0.2)
            
        # 2. Then wait for the is_playing flag to go to False
        # This handles the case where the queue is empty but the last chunk is still playing
        while self.is_playing and time.time() - start_time < timeout:
            time.sleep(0.3)
            
        # 3. Add a guaranteed buffer wait time to ensure audio has completely finished
        # This handles the case where threads might not have properly updated the is_playing flag
        # Use shorter buffer for short audio, longer for longer audio
        buffer_wait = min(1.0, self.total_audio_duration * 0.1)
        print(f"Adding buffer wait of {buffer_wait:.1f} seconds to ensure playback is complete...")
        time.sleep(buffer_wait)
            
        if time.time() - start_time >= timeout:
            print(f"Warning: Timed out waiting for audio playback to complete after {timeout:.1f}s")
            self.stop_playback()
            # Even after forcing stop, wait a moment to ensure resources are released
            time.sleep(0.5)
            # Reset tracking variables
            self.total_audio_duration = 0.0
            return False
        
        self.is_playing = False  # Ensure flag is reset
        
        # Reset tracking variables after successful playback
        self.total_audio_duration = 0.0
        return True

    def start_playback_thread(self, sample_rate):
        """Start the background playback thread if it's not already running."""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.should_stop_playback.clear()
            self.playback_thread = threading.Thread(
                target=self._audio_playback_thread,
                daemon=True
            )
            self.playback_thread.start()
    
    def stop_playback(self):
        """Improved playback stopping with buffer clearing"""
        if self.is_playing:
            self.should_stop_playback.set()
            
            # Clear queue but allow current chunk to finish
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
                    
            # Add a short silence to flush the audio buffer
            try:
                self.audio_queue.put((np.zeros(int(0.1*24000)), 24000))
            except:
                pass
                
            # Wait a short time for playback to complete
            time.sleep(0.1)
                
            self.is_playing = False
            
            # Reset duration tracking when stopping playback
            self.total_audio_duration = 0.0

    def _audio_playback_thread(self):
        """Background thread that plays audio fragments as they become available."""
        stream = None
        try:
            while not self.should_stop_playback.is_set():
                try:
                    # Get audio from queue with a timeout
                    audio_data, sample_rate = self.audio_queue.get(timeout=0.5)
                    
                    # Set playing flag to true
                    self.is_playing = True
                    
                    # Check if we should stop
                    if self.should_stop_playback.is_set():
                        self.audio_queue.task_done()
                        break
                    
                    # Create or recreate stream if needed
                    if stream is None:
                        stream = self.pyaudio.open(
                            format=pyaudio.paFloat32,
                            channels=1,
                            rate=sample_rate,
                            output=True
                        )
                    
                    # Calculate approximate playback time for this chunk
                    playback_duration = len(audio_data) / sample_rate
                    playback_start = time.time()
                    
                    # Play audio
                    try:
                        stream.write(audio_data.tobytes())
                        # Update tracking - subtract the duration of audio just played
                        self.total_audio_duration = max(0.0, self.total_audio_duration - playback_duration)
                    except Exception as e:
                        print(f"Error writing to audio stream: {e}")
                    finally:
                        self.audio_queue.task_done()
                        
                        # Log actual playback time vs expected
                        actual_duration = time.time() - playback_start
                        if abs(actual_duration - playback_duration) > 0.2:  # Only log if significantly different
                            print(f"Audio timing: expected={playback_duration:.2f}s, actual={actual_duration:.2f}s")
                    
                except queue.Empty:
                    # No audio in queue, set playing flag to false if queue is empty
                    if self.audio_queue.empty():
                        self.is_playing = False
                        # Reset duration tracking when queue is emptied
                        if self.total_audio_duration > 0.1:  # Only reset if there's significant remaining duration
                            print(f"Queue emptied with {self.total_audio_duration:.2f}s of tracked audio remaining, resetting")
                            self.total_audio_duration = 0.0
                    continue
                except Exception as e:
                    print(f"Error in audio playback thread: {e}")
                    if self.audio_queue.unfinished_tasks > 0:
                        try:
                            self.audio_queue.task_done()
                        except:
                            pass
        finally:
            # Clean up
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"Error closing audio stream: {e}")
            self.is_playing = False
            self.total_audio_duration = 0.0  # Reset duration tracking


    def __del__(self):
        """Cleanup PyAudio resources."""
        try:
            self.stop_playback()
            
            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except:
                    break
                    
            # Wait for thread to terminate safely
            if self.playback_thread and self.playback_thread.is_alive():
                try:
                    self.playback_thread.join(timeout=0.5)
                except:
                    pass
            
            # Finally terminate PyAudio
            if hasattr(self, 'pyaudio'):
                try:
                    self.pyaudio.terminate()
                except:
                    pass
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Continue with cleanup despite errors 