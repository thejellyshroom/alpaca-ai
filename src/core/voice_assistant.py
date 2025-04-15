from ..utils.audio_handler import AudioHandler
from ..utils.stt_handler import Transcriber
from ..utils.llm_handler import LLMHandler
from ..utils.tts_handler import TTSHandler
from ..utils.helper_functions import *
import os
import torch
import time
import traceback
import threading

#TODO: clean up code and split into multiple files/functions
class VoiceAssistant:
    def __init__(self, 
                 # Configuration dictionaries
                 asr_config=None,
                 tts_config=None,
                 llm_config=None):
        # Store configuration dictionaries
        self.asr_config = asr_config or {}
        self.tts_config = tts_config or {}
        self.llm_config = llm_config or {}
        
        # Extract key configuration parameters
        self.tts_model = self.tts_config.get('model_id')
        kokoro_conf = self.tts_config.get('kokoro', {})
        self.tts_voice = kokoro_conf.get('voice')
        self.speed = kokoro_conf.get('speed')
        self.expressiveness = kokoro_conf.get('expressiveness')
        self.variability = kokoro_conf.get('variability')
        self.transcription_model = self.asr_config.get('model_id')
        
        local_conf = self.llm_config.get('local', {})
        self.temperature = local_conf.get('temperature')
        self.top_p = local_conf.get('top_p')
        self.top_k = local_conf.get('top_k')
        
        # Component placeholders
        self.audio_handler = None
        self.transcriber = None
        self.llm_handler = None
        self.tts_handler = None
        self.tts_enabled = False
        
        # Conversation history
        self.conversation_history = [
            {"role": "system", "content": "You are a severely unhelpful AI that is a menace to the world."}
        ]
        
        # Initialize components
        self.load_all_components()
        
    def load_all_components(self):
        """Load all components of the voice assistant."""
        self.load_audio_handler()
        self.load_transcriber()
        self.load_llm_handler()
        self.load_tts_handler()
        
        # Print initialization information
        print("\nAI Voice Assistant initialized!")
        print(f"Transcription model: {self.transcription_model}")
        
        # Display device information
        if self.transcriber and hasattr(self.transcriber, 'device'):
            device = "CUDA" if torch.cuda.is_available() else "CPU"
        else:
            device = getattr(self.transcriber, 'device', 'unknown')
        print(f"Device: {device}")
        
        print(f"TTS Model: {self.tts_model}")
        print(f"TTS Voice: {self.tts_voice}")
        print(f"Speech Speed: {self.speed}x")
    
    def load_audio_handler(self):
        """Load the audio handler component."""
        self.audio_handler = unload_component(self.audio_handler, "audio_handler")
        try:
            self.audio_handler = AudioHandler(config=self.asr_config)
            print("Audio handler initialized successfully.")
        except Exception as e:
            print(f"Error initializing audio handler: {str(e)}")
    
    def load_transcriber(self):
        """Load the transcription component."""
        self.transcriber = unload_component(self.transcriber, "transcriber")
        try:
            model_id = self.transcription_model
            print(f"Initializing transcriber with model: {model_id}")
            self.transcriber = Transcriber(config=self.asr_config)
            print("Transcriber initialized successfully.")
        except Exception as e:
            print(f"Error initializing transcriber: {str(e)}")
            traceback.print_exc()
            self.transcriber = None
    
    def load_llm_handler(self):
        """Load the LLM handler."""
        print("Loading LLM handler...")
        if self.llm_handler is None:
            self.llm_handler = LLMHandler(config=self.llm_config)
        
    def load_tts_handler(self):
        """Load the TTS handler."""
        print("Loading TTS handler...")
        self.tts_handler = unload_component(self.tts_handler, "tts_handler")
        
        try:
            self.tts_handler = TTSHandler(self.tts_config)
            self.tts_enabled = True
            print("TTS handler loaded successfully.")
        except Exception as e:
            print(f"Error loading TTS handler: {e}")
            self.tts_enabled = False
        
    def listen(self, duration=None, timeout=None):
        """Record audio and transcribe it.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds
            timeout (int, optional): Maximum seconds to wait before giving up
            
        Returns:
            str: Transcribed text or error code
        """
        try:
            # Ensure any ongoing playback is stopped
            if self.audio_handler:
                if self.audio_handler.is_playing:
                    print("Stopping any ongoing audio playback before listening...")
                    self.audio_handler.stop_playback()
                    self.audio_handler.wait_for_playback_complete()
                
                print("Starting new listening session...")
                audio_file = self.audio_handler.listen_for_speech(
                    timeout=timeout,
                    stop_playback=True
                )
                
            # Return early for timeout errors
            if audio_file == "low_energy":
                print("Timeout error detected: speech volume too low")
                return "low_energy"
            elif audio_file == "TIMEOUT_ERROR":
                print("Timeout error detected: no speech detected within 5 seconds")
                return "TIMEOUT_ERROR"
                
            # Transcribe the audio
            try:    
                transcribed_text = self.transcriber.transcribe(audio_file)
                print(f"Transcription successful: {len(transcribed_text)} characters")
                return transcribed_text
            except Exception as e:
                print(f"Error during transcription: {str(e)}")
                traceback.print_exc()
                return ""
        except Exception as e:
            print(f"Unexpected error in listen method: {str(e)}")
            traceback.print_exc()
            return ""
    
    def process_and_respond(self, text):
        """Process text with LLM and get response.
        
        Args:
            text (list): Conversation history list
            
        Returns:
            str or generator: LLM's response (can be a generator for streaming)
        """
        query = text[-1]
        # Use RAG for knowledge-based queries
        if should_use_rag(query["content"]):
            response = self.llm_handler.get_rag_response(query["content"], self.conversation_history[:-1])
        else:
            response = self.llm_handler.get_response(self.conversation_history)
        return response
    
    def speak(self, text):
        """Convert text to speech and play it, allowing for interruption.

        Args:
            text (str): Text to speak

        Returns:
            str: "COMPLETED" if playback finished, "INTERRUPTED" if user spoke during playback.
        """
        if not self.tts_enabled:
            print("TTS is disabled. Cannot speak the response.")
            return "COMPLETED"

        try:
            if text is None:
                print("Warning: Received None text to speak")
                return "COMPLETED"

            # Prepare for interruptible playback
            interrupt_event = threading.Event()
            interrupted = False

            # Start listening for interruptions
            if self.audio_handler:
                self.audio_handler.start_interrupt_listener(interrupt_event)

            # Split text into sentences for potentially better interrupt points
            sentences = split_into_sentences(text)

            # Play sentence by sentence, checking for interrupt
            for sentence in sentences:
                if interrupt_event.is_set():
                    print("Interrupt detected during sentence processing.")
                    interrupted = True
                    break

                # Synthesize sentence
                audio_array, sample_rate = self.tts_handler.synthesize(sentence)

                if interrupt_event.is_set():
                    print("Interrupt detected after sentence synthesis.")
                    interrupted = True
                    break

                if audio_array is not None and len(audio_array) > 0:
                    # Play audio (non-blocking)
                    self.audio_handler.play_audio(audio_array, sample_rate)
                    # Brief sleep allows checking interrupt event more frequently
                    # and gives playback thread time to start
                    time.sleep(0.1)
                else:
                    print("Generated audio is empty. Skipping sentence.")

            # If already interrupted, skip waiting for remaining audio
            if not interrupted:
                # Wait for any remaining queued audio to play, checking for interrupts
                print("Waiting for final audio chunks to play...")
                while self.audio_handler and self.audio_handler.is_playing:
                    if interrupt_event.is_set():
                        print("Interrupt detected while waiting for playback completion.")
                        interrupted = True
                        break
                    time.sleep(0.1)

            # --- Cleanup --- 
            if interrupted:
                print("Stopping playback due to interrupt.")
                if self.audio_handler:
                    # stop_playback also stops the listener
                    self.audio_handler.stop_playback()
                return "INTERRUPTED"
            else:
                print("Playback completed normally.")
                # Ensure listener is stopped if playback finished naturally
                # (It should be stopped by the playback thread finishing, but just in case)
                if self.audio_handler:
                    self.audio_handler.stop_interrupt_listener()
                return "COMPLETED"

        except Exception as e:
            print(f"Error in speak method: {str(e)}")
            traceback.print_exc()
            # Ensure listener is stopped on error
            if self.audio_handler:
                 self.audio_handler.stop_interrupt_listener()
            return "ERROR"

    def interaction_loop(self, duration=None, timeout=10, phrase_limit=10):
        """Record audio, transcribe, process with streaming response, handle interrupts.

        Args:
            duration (int, optional): Fixed recording duration in seconds
            timeout (int, optional): Maximum seconds to wait before giving up
            phrase_limit (int, optional): Maximum seconds for a single phrase

        Returns:
            tuple: (transcribed_text, ai_response)
        """
        try:
            # Ensure all audio playback is stopped before starting loop iteration
            if self.audio_handler:
                self.audio_handler.stop_playback()

            # Listen for user input
            print("\nListening for your voice...")
            transcribed_text = self.listen(duration=duration, timeout=timeout)

            # Handle specific error codes from listen
            if transcribed_text in ["TIMEOUT_ERROR", "low_energy"]:
                ai_response = f"Seems like you didn't say anything or it was too quiet ({transcribed_text})."
                print("\nAssistant:", ai_response)
                # Optionally speak this message, or just print and loop
                # self.speak(ai_response) # Consider if you want assistant to speak timeout messages
                return "", ai_response

            # Handle general lack of transcription
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                # Assuming empty string or None might indicate other listen errors
                print("\nAssistant: I didn't catch that. Could you please repeat?")
                # self.speak("I didn't catch that. Could you please repeat?")
                return "", "No transcription received."

            # --- Process valid input ---
            print(f"\nYou: {transcribed_text}")
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": transcribed_text})

            # Get response from LLM
            print("\nAssistant thinking...")
            response_source = self.process_and_respond(self.conversation_history)

            # --- Consume generator if necessary --- 
            if hasattr(response_source, '__iter__') and not isinstance(response_source, str):
                print("(Streaming response detected, collecting...)")
                ai_response_parts = []
                for chunk in response_source:
                    # Assuming chunks are strings, handle potential errors if not
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True) # Print stream progress
                        ai_response_parts.append(chunk)
                    else:
                         print(f"\nWarning: Received non-string chunk from LLM: {type(chunk)}")
                ai_response = "".join(ai_response_parts)
                print() # Newline after streaming
            else:
                # Assume it's already a string
                ai_response = response_source

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": ai_response})

            # Print and speak the response (ai_response is now guaranteed to be a string)
            print(f"\nAssistant: {ai_response}")
            speak_status = self.speak(ai_response)

            # Check if speaking was interrupted
            if speak_status == "INTERRUPTED":
                print("(Assistant speech interrupted by user)")
                # Loop immediately back to listening
                return "INTERRUPTED", ai_response
            elif speak_status == "ERROR":
                print("(An error occurred during speech synthesis or playback)")
                # Decide how to proceed, maybe retry or just loop

            # Return normally if completed
            return transcribed_text, ai_response

        except Exception as e:
            print(f"Error in interaction loop: {str(e)}")
            traceback.print_exc()
            # Ensure playback/listening is stopped on unexpected loop errors
            if self.audio_handler:
                self.audio_handler.stop_playback()
            return "ERROR", str(e)

    def main_loop(self, duration=None, timeout=5, phrase_limit=10):
        """The main loop that calls interaction_loop repeatedly."""
        try:
            while True:
                result, _ = self.interaction_loop(
                    duration=duration,
                    timeout=timeout,
                    phrase_limit=phrase_limit
                )
                # If interrupted, the loop continues automatically.
                # If an error occurred, maybe add a delay or specific handling.
                if result == "ERROR":
                    print("Recovering from interaction error...")
                    time.sleep(2)

        except KeyboardInterrupt:
            print("\nExiting Voice Assistant...")
        finally:
            # Cleanup resources
            if self.audio_handler:
                self.audio_handler.stop_playback() # Ensure everything stops
            print("Cleanup complete.")
            