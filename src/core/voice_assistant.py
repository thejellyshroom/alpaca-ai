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
import types # Needed to check for generator type
import gc # Needed for garbage collection

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
        self.load_stt()
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
    
    def load_stt(self):
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
            # Directly return the generator from get_rag_response (which now yields strings)
            response = self.llm_handler.get_rag_response(query["content"], self.conversation_history[:-1])
        else:
            # This already returns the correct generator yielding strings
            response = self.llm_handler.get_response(self.conversation_history)
        return response
    
    def speak(self, response_source):
        """Convert text (string or generator) to speech and play it, allowing for interruption.

        Args:
            response_source (str or generator): Text to speak or generator yielding text chunks.

        Returns:
            tuple: (status, full_text) where status is "COMPLETED" or "INTERRUPTED",
                   and full_text is the complete response string.
                   Returns ("DISABLED", "") if TTS is disabled.
                   Returns ("ERROR", error_message) on failure.
        """
        if not self.tts_enabled or not self.audio_handler or not self.tts_handler:
            print("TTS is disabled or handlers not available. Cannot speak.")
            return ("DISABLED", "") # Return status and empty text

        interrupt_event = threading.Event()
        interrupted = False
        full_response_text = ""
        tts_buffer = ""
        min_chunk_len = 50 # Minimum characters for *subsequent* chunks
        sentence_ends = (".", "!", "?", "\n")
        initial_words_spoken = False # Flag for initial fast chunk
        word_count = 0 # Word counter for initial chunk
        approx_words_for_initial_chunk = 8 # Target words for initial chunk

        try:
            print("Assistant:", end="", flush=True) # Print prompt for assistant response

            # Start listening for interruptions *before* processing/playing
            self.audio_handler.detector.start_interrupt_listener(interrupt_event)

            # Handle generator (streaming) case
            if isinstance(response_source, types.GeneratorType):
                for token in response_source:
                    if interrupt_event.is_set():
                        print("\nInterrupt detected during response generation.")
                        interrupted = True
                        break

                    print(token, end="", flush=True) # Print token immediately
                    full_response_text += token
                    tts_buffer += token

                    # --- Logic for Speaking Chunks --- #
                    speak_this_chunk = False

                    if not initial_words_spoken:
                        # Estimate word count (simple space counting)
                        word_count = tts_buffer.count(' ') + 1 
                        # Check for initial chunk condition (enough words OR first sentence end)
                        if word_count >= approx_words_for_initial_chunk or any(tts_buffer.endswith(punc) for punc in sentence_ends):
                            speak_this_chunk = True
                            initial_words_spoken = True # Mark initial chunk as handled
                            print(f" (Speaking initial chunk: {word_count} words)", end="", flush=True) # Debug
                    else:
                        # Regular chunking logic after initial chunk
                        buffer_len = len(tts_buffer)
                        if buffer_len >= min_chunk_len and any(tts_buffer.endswith(punc) for punc in sentence_ends):
                             speak_this_chunk = True
                             # print(" (Speaking regular chunk)", end="", flush=True) # Debug

                    # --- Synthesize and Play if a chunk is ready --- #
                    if speak_this_chunk and tts_buffer.strip():
                        chunk_to_speak = tts_buffer.strip()
                        tts_buffer = "" # Clear buffer *before* potentially long synthesis/playback
                        word_count = 0 # Reset word count after speaking a chunk

                        try:
                            # Synthesize the chunk
                            audio_array, sample_rate = self.tts_handler.synthesize(chunk_to_speak)

                            if interrupt_event.is_set():
                                print("\nInterrupt detected after chunk synthesis.")
                                interrupted = True
                                break

                            # Play the synthesized audio
                            if audio_array is not None and len(audio_array) > 0:
                                self.audio_handler.player.play_audio(audio_array, sample_rate)
                                # Short sleep allows interrupt check and prevents busy-wait
                                time.sleep(0.05)
                            else:
                                print(f"\nWarning: Generated audio for chunk '{chunk_to_speak}' is empty.")

                        except Exception as e:
                             print(f"\nError during TTS synthesis for chunk '{chunk_to_speak}': {e}")
                             # Continue processing next tokens even if one chunk fails
                             time.sleep(0.1)

                # End of generator loop
                print() # Add newline after streaming output

                # Synthesize any remaining text in the buffer after generator finishes
                if not interrupted and tts_buffer.strip():
                     final_chunk = tts_buffer.strip()
                     print(f"(Synthesizing remaining: '{final_chunk}')")
                     try:
                         audio_array, sample_rate = self.tts_handler.synthesize(final_chunk)
                         if audio_array is not None and len(audio_array) > 0:
                             self.audio_handler.player.play_audio(audio_array, sample_rate)
                     except Exception as e:
                         print(f"\nError synthesizing final text segment: {e}")

            # Handle string (non-streaming) case
            elif isinstance(response_source, str):
                full_response_text = response_source # Already have the full text
                print(full_response_text) # Print the whole response at once

                sentences = split_into_sentences(full_response_text)
                for sentence in sentences:
                    if interrupt_event.is_set():
                        print("Interrupt detected during sentence processing.")
                        interrupted = True
                        break

                    try:
                        audio_array, sample_rate = self.tts_handler.synthesize(sentence)
                        if interrupt_event.is_set():
                            print("Interrupt detected after sentence synthesis.")
                            interrupted = True
                            break

                        if audio_array is not None and len(audio_array) > 0:
                            self.audio_handler.player.play_audio(audio_array, sample_rate)
                            # Need to wait slightly for playback buffer, allows interrupt check
                            time.sleep(0.1)
                        else:
                             print("Generated audio is empty for sentence. Skipping.")
                    except Exception as e:
                        print(f"\nError synthesizing sentence: {e}")
                        time.sleep(0.1)

            else:
                # Handle unexpected input type
                print(f"\nError: speak method received unexpected type: {type(response_source)}")
                self.audio_handler.detector.stop_interrupt_listener() # Stop listener if started
                return ("ERROR", f"Unexpected response type: {type(response_source)}")


            # --- Wait for Playback Completion (if not already interrupted) ---
            if not interrupted:
                print("Waiting for audio playback to complete...")
                wait_start_time = time.time()
                # Wait while playing, checking interrupt event frequently
                while self.audio_handler.player.is_playing:
                    if interrupt_event.is_set():
                        print("\nInterrupt detected while waiting for playback completion.")
                        interrupted = True
                        break
                    # Check for excessive wait time (e.g., > 60 seconds)
                    if time.time() - wait_start_time > 60:
                         print("\nWarning: Playback wait timed out (> 60s). Forcing stop.")
                         interrupted = True # Treat as interruption/error
                         break
                    time.sleep(0.1) # Check interrupt status roughly 10 times/sec

            # --- Cleanup and Return Status ---
            if interrupted:
                print("Stopping playback due to interrupt.")
                self.audio_handler.stop_playback() # Stops player AND detector
                return ("INTERRUPTED", full_response_text)
            else:
                print("Playback completed.")
                self.audio_handler.detector.stop_interrupt_listener() # Stop only detector
                # Ensure player is fully stopped and cleaned (redundant but safe)
                self.audio_handler.player.wait_for_playback_complete(timeout=5)
                return ("COMPLETED", full_response_text)

        except Exception as e:
            print(f"\nError in speak method: {e}")
            traceback.print_exc()
            # Ensure cleanup happens on error
            if self.audio_handler:
                 self.audio_handler.stop_playback() # Stops player and detector
            return ("ERROR", str(e)) # Return error status and message

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
                ai_response_text = f"Seems like you didn't say anything or it was too quiet ({transcribed_text})."
                print("\nAssistant:", ai_response_text)
                # Optionally speak this message (but it won't be streamed)
                # self.speak(ai_response_text)
                return "", ai_response_text # Return empty user text, assistant message

            # Handle general lack of transcription
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                ai_response_text = "I didn't catch that. Could you please repeat?"
                print("\nAssistant:", ai_response_text)
                # self.speak(ai_response_text)
                return "", ai_response_text

            # --- Process valid input ---
            print(f"\nYou: {transcribed_text}")
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": transcribed_text})

            # Get response from LLM (potentially a generator)
            print("\nAssistant thinking...")
            response_source = self.process_and_respond(self.conversation_history)

            # --- Speak the response (handles streaming internally) ---
            # speak now returns (status, full_text)
            speak_status, ai_response_text = self.speak(response_source)

            # Add assistant response (full text) to history *after* speak finishes/is interrupted
            if ai_response_text: # Only add if we got some text back
                self.conversation_history.append({"role": "assistant", "content": ai_response_text})

            # Handle speak status
            if speak_status == "INTERRUPTED":
                print("(Assistant speech interrupted by user)")
                # Loop immediately back to listening
                return "INTERRUPTED", ai_response_text # Indicate interrupt, return partial text
            elif speak_status == "ERROR":
                print("(An error occurred during speech synthesis or playback)")
                # Return error status, potentially the error message from speak
                return "ERROR", ai_response_text
            elif speak_status == "DISABLED":
                 print("(TTS is disabled, assistant response not spoken)")
                 # Continue normally, but without speech

            # Return normally if completed (speak_status == "COMPLETED")
            return transcribed_text, ai_response_text

        except Exception as e:
            print(f"\nError in interaction loop: {str(e)}")
            traceback.print_exc()
            # Ensure playback/listening is stopped on unexpected loop errors
            if self.audio_handler:
                self.audio_handler.stop_playback()
            return "ERROR", str(e)

    def main_loop(self, duration=None, timeout=5, phrase_limit=10):
        """The main loop that calls interaction_loop repeatedly."""
        try:
            while True:
                # interaction_loop now returns (user_text_or_status, assistant_text_or_error)
                user_input_status, assistant_output = self.interaction_loop(
                    duration=duration,
                    timeout=timeout,
                    phrase_limit=phrase_limit
                )
                # If interrupted, the loop continues automatically.
                # If an error occurred in interaction_loop, maybe add a delay or specific handling.
                if user_input_status == "ERROR":
                    print(f"Recovering from interaction error: {assistant_output}")
                    time.sleep(2)
                elif user_input_status == "INTERRUPTED":
                     print("Interaction interrupted, starting new loop.")
                     # Continue immediately

        except KeyboardInterrupt:
             print("\nExiting Voice Assistant...")
        finally:
            # Cleanup resources
            print("Performing final cleanup...")
            if self.audio_handler:
                self.audio_handler.stop_playback() # Ensure everything stops
            # Explicitly delete handlers to trigger __del__ for cleanup before exit
            if hasattr(self, 'tts_handler'): del self.tts_handler
            if hasattr(self, 'llm_handler'): del self.llm_handler
            if hasattr(self, 'transcriber'): del self.transcriber
            if hasattr(self, 'audio_handler'): del self.audio_handler
            gc.collect() # Suggest garbage collection
            print("Cleanup complete.")
            