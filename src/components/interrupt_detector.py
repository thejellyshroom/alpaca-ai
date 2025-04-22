import pyaudio
import threading
import time
import numpy as np
import math
import torch

class InterruptDetector:
    def __init__(self, pyaudio_instance, config):
        self.pyaudio = pyaudio_instance
        self.config = config

        # --- Silero VAD Setup ---
        try:
            # Use force_reload=True if you want to ensure the latest version
            self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
            (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.vad_utils
            self.vad_model.eval() # Set model to evaluation mode
            print("Silero VAD model loaded successfully.")
        except Exception as e:
            print(f"Error loading Silero VAD model: {e}. Interrupt detection might be less accurate.")
            self.vad_model = None
            self.vad_utils = None

        # --- Audio Stream Parameters ---
        # VAD model expects 16kHz, mono
        self.vad_sample_rate = 16000
        self.channels = 1 # Force mono for VAD processing
        # Silero VAD requires specific chunk sizes: 512 samples for 16kHz
        self.chunk = 512 
        self.format = pyaudio.paInt16 # VAD works on int16

        # --- VAD/Energy Parameters from Config ---
        # Use get with defaults for robustness
        vad_audio_validation = config.get('audio_validation', {}) # Get sub-dict
        self.vad_energy_threshold = vad_audio_validation.get('vad_energy_threshold', 300)
        self.vad_activation_chunks = vad_audio_validation.get('vad_activation_chunks', 3)
        self.vad_confidence_threshold = vad_audio_validation.get('vad_confidence_threshold', 0.5) # Confidence threshold for VAD

        # Threading and state
        self.interrupt_listener_thread = None
        self.should_stop_interrupt_listener = threading.Event()
        self._interrupt_event_ref = None # Reference to the caller's event

    def _calculate_rms(self, data):
        """Calculate Root Mean Square of audio data (int16 numpy array)."""
        try:
            data = data.astype(np.int16)
        except ValueError:
            print("Warning: Could not convert audio data to int16 for RMS calculation.")
            return 0
        if data.size == 0:
            return 0
        # Use float64 for intermediate calculation to avoid overflow
        rms = math.sqrt(np.mean(np.square(data.astype(np.float64))))
        return rms

    def _process_vad(self, audio_chunk_int16):
        """Process an audio chunk with Silero VAD."""
        if not self.vad_model:
            return 0.0 # Return 0 confidence if VAD model failed to load

        try:
            # Convert int16 numpy array to float32 tensor expected by VAD
            audio_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32)

            # Get speech probability
            speech_prob = self.vad_model(audio_tensor, self.vad_sample_rate).item()
            return speech_prob
        except Exception as e:
            print(f"Error during VAD processing: {e}")
            return 0.0 # Return 0 confidence on error

    def _interrupt_listener_run(self):
        """Background thread to listen for user interruption via VAD and Energy."""
        print(f"Interrupt listener started (Rate: {self.vad_sample_rate} Hz, Chunk: {self.chunk}, Channels: {self.channels}).")
        stream = None
        active_chunks = 0
        try:
            stream = self.pyaudio.open(format=self.format, # paInt16
                                       channels=self.channels, # Mono
                                       rate=self.vad_sample_rate, # 16000 Hz
                                       input=True,
                                       frames_per_buffer=self.chunk)

            while not self.should_stop_interrupt_listener.is_set():
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    # Get int16 numpy array from buffer
                    audio_chunk_int16 = np.frombuffer(data, dtype=np.int16)

                    rms = self._calculate_rms(audio_chunk_int16)

                    speech_prob = self._process_vad(audio_chunk_int16)

                    is_speech_detected = (rms > self.vad_energy_threshold and speech_prob >= self.vad_confidence_threshold)

                    if is_speech_detected:
                        active_chunks += 1
                    else:
                        # Decay activation counter if no speech detected
                        active_chunks = max(0, active_chunks - 1)

                    if active_chunks >= self.vad_activation_chunks:
                        print(f"Interrupt detected! (RMS: {rms:.2f}, VAD: {speech_prob:.3f} >= {self.vad_confidence_threshold} for {active_chunks} chunks)")
                        if self._interrupt_event_ref and not self._interrupt_event_ref.is_set():
                            self._interrupt_event_ref.set() # Signal the main thread/caller

                        # Reset active chunks slightly to require sustained sound and prevent rapid re-triggering
                        active_chunks = self.vad_activation_chunks - 1

                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        print("Input overflowed in interrupt listener. Skipping chunk.")
                    else:
                        print(f"IOError in interrupt listener: {e}")
                        # Avoid busy-waiting on persistent IOErrors
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Error in interrupt listener loop: {e}")
                    time.sleep(0.1)

        except Exception as e:
             print(f"Error setting up interrupt listener stream: {e}")

        finally:
            # --- Stream Cleanup --- #
            if stream is not None:
                try:
                    if stream.is_active():
                       stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"Error closing interrupt listener stream: {e}")
            # --- Final State Update --- #
            self.should_stop_interrupt_listener.set()

    # --- Public Methods (start/stop/cleanup remain largely the same) ---

    def start_interrupt_listener(self, interrupt_event):
        """Starts the VAD interrupt listener thread."""
        if self.interrupt_listener_thread is None or not self.interrupt_listener_thread.is_alive():
            # Ensure VAD model is available before starting
            if not self.vad_model:
                print("Cannot start interrupt listener: Silero VAD model not loaded.")
                return

            self.should_stop_interrupt_listener.clear()
            self._interrupt_event_ref = interrupt_event # Store reference to the event
            self.interrupt_listener_thread = threading.Thread(
                target=self._interrupt_listener_run,
                daemon=True
            )
            self.interrupt_listener_thread.start()
        else:
            print("Interrupt listener already running.")

    def stop_interrupt_listener(self):
        """Stops the VAD interrupt listener thread by setting an event."""
        if self.interrupt_listener_thread and self.interrupt_listener_thread.is_alive():
             if not self.should_stop_interrupt_listener.is_set():
                self.should_stop_interrupt_listener.set()
             else:
                 pass
        self.interrupt_listener_thread = None
        self._interrupt_event_ref = None

    def cleanup(self):
        """Clean up resources, stop thread."""
        self.stop_interrupt_listener()
        print("InterruptDetector cleanup finished.") 