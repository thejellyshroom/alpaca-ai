import pyaudio
import queue
import threading
import time
import numpy as np

class AudioPlayer:
    def __init__(self, pyaudio_instance, default_sample_rate=22050):
        self.pyaudio = pyaudio_instance
        self.default_sample_rate = default_sample_rate
        self.audio_queue = queue.Queue(maxsize=100)
        self.playback_thread = None
        self.is_playing = False
        self.should_stop_playback = threading.Event()
        self.total_audio_duration = 0.0
        self.last_audio_timestamp = 0.0

    def play_audio(self, audio_data, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.default_sample_rate

        # Convert PyTorch tensor to numpy array if needed
        if hasattr(audio_data, 'detach') and hasattr(audio_data, 'cpu') and hasattr(audio_data, 'numpy'):
            audio_data = audio_data.detach().cpu().numpy()

        # Ensure audio data is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        audio_duration = len(audio_data) / sample_rate

        self.total_audio_duration += audio_duration
        self.last_audio_timestamp = time.time()

        self.start_playback_thread(sample_rate)

        # Add audio to the queue
        self.audio_queue.put((audio_data, sample_rate))
        self.is_playing = True

    def start_playback_thread(self, sample_rate):
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.should_stop_playback.clear()
            self.playback_thread = threading.Thread(
                target=self._audio_playback_thread,
                daemon=True
            )
            self.playback_thread.start()

    def _audio_playback_thread(self):
        stream = None
        current_sample_rate = self.default_sample_rate # Initialize with default
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

                    # Create or recreate stream if needed or if sample rate changes
                    if stream is None or sample_rate != current_sample_rate:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                        stream = self.pyaudio.open(
                            format=pyaudio.paFloat32,
                            channels=1, # Assuming mono, make configurable if needed
                            rate=sample_rate,
                            output=True
                        )
                        current_sample_rate = sample_rate

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


                except queue.Empty:
                    # No audio in queue, set playing flag to false if queue is empty
                    if self.audio_queue.empty():
                        self.is_playing = False
                        # Reset duration tracking when queue is emptied
                        if self.total_audio_duration > 0.1: # Only reset if there's significant remaining duration
                            print(f"Queue emptied with {self.total_audio_duration:.2f}s of tracked audio remaining, resetting")
                            self.total_audio_duration = 0.0
                    continue
                except Exception as e:
                    print(f"Error in audio playback thread: {e}")
                    if self.audio_queue.unfinished_tasks > 0:
                        try:
                            self.audio_queue.task_done()
                        except ValueError: # Can happen if task_done() called too many times
                             pass

        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"Error closing audio stream: {e}")
            self.is_playing = False
            self.total_audio_duration = 0.0 # Reset duration tracking

    def wait_for_playback_complete(self, timeout=None):
        """Wait until all audio playback has completed.

        Args:
            timeout (float, optional): Maximum time to wait in seconds. If None, calculated dynamically.

        Returns:
            bool: True if playback completed, False if timed out
        """
        if not self.is_playing and self.audio_queue.empty():
            self.total_audio_duration = 0.0 # Ensure reset if already finished
            return True

        # Calculate dynamic timeout if not provided
        if timeout is None:
            # Base timeout on total audio duration plus buffer
            # Formula: audio_duration * 1.5 + 2.0 seconds (min 5 seconds, max 60 seconds)
            estimated_remaining = self.total_audio_duration
            timeout = min(max(estimated_remaining * 1.5 + 2.0, 5.0), 60.0)

        print(f"Waiting for audio playback to complete (timeout: {timeout:.1f}s, estimated remaining: {estimated_remaining:.1f}s)...")

        start_time = time.time()

        # Wait for the queue to be empty
        queue_empty_time = None
        while not self.audio_queue.empty() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        if self.audio_queue.empty():
             queue_empty_time = time.time()

        # Then wait for the is_playing flag to go to False
        while self.is_playing and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Add a small buffer wait after is_playing is false, ensures the last chunk finishes
        if not self.is_playing:
             buffer_wait_start = time.time()
             buffer_wait_duration = min(1.0, estimated_remaining * 0.1 + 0.2) # Dynamic buffer
             while (time.time() - buffer_wait_start) < buffer_wait_duration and (time.time() - start_time) < timeout:
                  time.sleep(0.05)
             print(f"Buffer wait complete ({buffer_wait_duration:.2f}s). Queue empty at: {queue_empty_time - start_time if queue_empty_time else 'N/A':.2f}s, Playing flag false at: {time.time() - buffer_wait_start - start_time:.2f}s")


        if time.time() - start_time >= timeout:
            print(f"Warning: Timed out waiting for audio playback to complete after {timeout:.1f}s")
            self.stop_playback(force=True) # Force stop on timeout
            return False

        self.is_playing = False # Ensure flag is reset
        self.total_audio_duration = 0.0 # Reset tracking variables after successful playback
        print("Audio playback complete.")
        return True

    def stop_playback(self, force=False):
        """Stop audio playback and clear the queue.

        Args:
            force (bool): If True, forcefully stops even if thread seems stuck.
        """
        if self.is_playing or force:
            self.should_stop_playback.set()

            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
                except ValueError: # task_done() called too many times
                    break

            # Optionally wait briefly for thread to stop
            if self.playback_thread and self.playback_thread.is_alive():
                 self.playback_thread.join(timeout=0.2) # Short wait

            self.is_playing = False
            self.total_audio_duration = 0.0
            print("Audio playback stopped.")

    def cleanup(self):
        """Clean up resources, stop thread."""
        self.stop_playback(force=True)
        if self.playback_thread and self.playback_thread.is_alive():
            try:
                self.playback_thread.join(timeout=0.5)
            except Exception as e:
                print(f"Error joining playback thread during cleanup: {e}")
        self.playback_thread = None
        print("AudioPlayer cleanup finished.")
