from ..components.audio_handler import AudioHandler
from ..components.stt_handler import Transcriber
from ..components.llm_handler import LLMHandler
from ..components.tts_handler import TTSHandler
from ..utils.helper_functions import *
from ..utils.conversation_manager import ConversationManager
from ..utils.component_manager import ComponentManager
import time
import traceback
import threading
import types # Needed to check for generator type
import gc # Needed for garbage collection

#TODO: clean up code and split into multiple files/functions
class VoiceAssistant:
    def __init__(self, 
                 asr_config=None,
                 tts_config=None,
                 llm_config=None,
                 duration=None, 
                 timeout=5, 
                 phrase_limit=10):
        
        # Configs are passed directly to ComponentManager
        asr_config = asr_config or {}
        tts_config = tts_config or {}
        llm_config = llm_config or {}
        
        # Extract system prompt (potentially move to ConfigLoader later)
        system_prompt = llm_config.get('system_prompt', "You are a helpful assistant.")
        self.conversation_manager = ConversationManager(system_prompt=system_prompt)

        # Initialize Component Manager (which loads handlers)
        self.component_manager = ComponentManager(asr_config, tts_config, llm_config)

        # Store loop parameters (originating from ConfigLoader -> main.py)
        self._duration_arg = duration
        self._timeout_arg = timeout
        self._phrase_limit_arg = phrase_limit
        
        print("\nAI Voice Assistant Core Initialized!")
        # Summary is now printed by ComponentManager
        
    def listen(self, duration=None, timeout=None):
        """Record audio and transcribe it using ComponentManager handlers."""
        audio_handler = self.component_manager.audio_handler
        transcriber = self.component_manager.transcriber
        
        if not audio_handler or not transcriber:
             print("Error: Audio Handler or Transcriber not available.")
             return "ERROR"

        try:
            # Ensure any ongoing playback is stopped
            if audio_handler.is_playing:
                print("Stopping any ongoing audio playback before listening...")
                audio_handler.stop_playback()
                # Use player directly for waiting
                audio_handler.player.wait_for_playback_complete(timeout=2.0)
            
            print("Starting new listening session...")
            audio_file = audio_handler.listen_for_speech(
                timeout=timeout,
                stop_playback=False # Already stopped above if needed
            )
                
            # Handle errors from listen_for_speech
            if audio_file in ["low_energy", "TIMEOUT_ERROR", None]:
                print(f"Listening failed or timed out: {audio_file}")
                return audio_file or "ERROR" # Return code or general error
                
            # Transcribe the audio
            print(f"Transcribing audio file: {audio_file}")
            transcribed_text = transcriber.transcribe(audio_file)
            if transcribed_text is None:
                 print("Transcription resulted in None.")
                 return "" # Treat as empty transcription
                 
            print(f"Transcription successful: {len(transcribed_text)} characters")
            return transcribed_text
            
        except AttributeError as e:
             print(f"Error accessing audio/transcriber components: {e}. Was initialization successful?")
             traceback.print_exc()
             return "ERROR"
        except Exception as e:
            print(f"Unexpected error in listen method: {str(e)}")
            traceback.print_exc()
            return "ERROR"
    
    def process_and_respond(self):
        """Process text with LLM using history and ComponentManager handlers."""
        llm_handler = self.component_manager.llm_handler
        if not llm_handler:
             print("Error: LLM Handler not available.")
             # Return an error message wrapped in a generator? Or handle differently?
             def error_gen(): yield "LLM Handler not available."
             return error_gen()
             
        current_history = self.conversation_manager.get_history()
        if not current_history:
            print("Error: Conversation history is empty.")
            def error_gen(): yield "Conversation history is empty."
            return error_gen()

        query_message = current_history[-1]
        query_content = query_message.get("content", "")
        previous_history = current_history[:-1]

        try:
            # Use RAG for knowledge-based queries
            if should_use_rag(query_content):
                response = llm_handler.get_rag_response(query_content, previous_history)
            else:
                response = llm_handler.get_response(current_history)
            return response
        except Exception as e:
            print(f"Error getting response from LLM Handler: {e}")
            traceback.print_exc()
            def error_gen(): yield f"Error communicating with LLM: {e}"
            return error_gen()
    
    def speak(self, response_source):
        """Convert text to speech using ComponentManager handlers."""
        tts_handler = self.component_manager.tts_handler
        audio_handler = self.component_manager.audio_handler
        tts_enabled = self.component_manager.tts_enabled

        if not tts_enabled or not audio_handler or not tts_handler:
            print("TTS is disabled or handlers not available. Cannot speak.")
            # Determine full text if possible, even if not speaking
            full_response_text = ""
            if isinstance(response_source, str):
                 full_response_text = response_source
            elif isinstance(response_source, types.GeneratorType):
                 try:
                      full_response_text = "".join(list(response_source))
                      print(f"Assistant (TTS Disabled): {full_response_text}")
                 except Exception as e:
                      print(f"Error consuming generator when TTS disabled: {e}")
            return ("DISABLED", full_response_text) 

        interrupt_event = threading.Event()
        interrupted = False
        full_response_text = ""
        tts_buffer = ""
        min_chunk_len = 50
        sentence_ends = (".", "!", "?", "\n")
        initial_words_spoken = False
        word_count = 0
        approx_words_for_initial_chunk = 8

        try:
            print("Assistant:", end="", flush=True)
            audio_handler.detector.start_interrupt_listener(interrupt_event)

            # Handle generator (streaming) case
            if isinstance(response_source, types.GeneratorType):
                for token in response_source:
                    if interrupt_event.is_set(): interrupted = True; break

                    print(token, end="", flush=True) 
                    full_response_text += token
                    tts_buffer += token

                    # Logic for Speaking Chunks
                    speak_this_chunk = False
                    if not initial_words_spoken:
                        word_count = tts_buffer.count(' ') + 1 
                        if word_count >= approx_words_for_initial_chunk or any(tts_buffer.endswith(punc) for punc in sentence_ends):
                            speak_this_chunk = True
                            initial_words_spoken = True
                    else:
                        buffer_len = len(tts_buffer)
                        if buffer_len >= min_chunk_len and any(tts_buffer.endswith(punc) for punc in sentence_ends):
                             speak_this_chunk = True

                    # Synthesize and Play if a chunk is ready
                    if speak_this_chunk and tts_buffer.strip():
                        chunk_to_speak = tts_buffer.strip()
                        tts_buffer = ""
                        word_count = 0
                        try:
                            audio_array, sample_rate = tts_handler.synthesize(chunk_to_speak)
                            if interrupt_event.is_set(): interrupted = True; break
                            if audio_array is not None and len(audio_array) > 0:
                                # Use player directly
                                audio_handler.player.play_audio(audio_array, sample_rate)
                                time.sleep(0.05)
                        except Exception as e:
                             print(f"\nError during TTS synthesis for chunk: {e}")
                             time.sleep(0.1)

                print() # Newline after streaming
                # Synthesize any remaining text in the buffer
                if not interrupted and tts_buffer.strip():
                     try:
                         audio_array, sample_rate = tts_handler.synthesize(tts_buffer.strip())
                         if audio_array is not None and len(audio_array) > 0:
                             audio_handler.player.play_audio(audio_array, sample_rate)
                     except Exception as e:
                         print(f"\nError synthesizing final text segment: {e}")
                         
            # Handle string (non-streaming) case
            elif isinstance(response_source, str):
                full_response_text = response_source 
                print(full_response_text)
                sentences = split_into_sentences(full_response_text)
                for sentence in sentences:
                    if interrupt_event.is_set(): interrupted = True; break
                    try:
                        audio_array, sample_rate = tts_handler.synthesize(sentence)
                        if interrupt_event.is_set(): interrupted = True; break
                        if audio_array is not None and len(audio_array) > 0:
                            audio_handler.player.play_audio(audio_array, sample_rate)
                            time.sleep(0.1)
                    except Exception as e:
                        print(f"\nError synthesizing sentence: {e}")
                        time.sleep(0.1)

            else:
                 print(f"\nError: speak method received unexpected type: {type(response_source)}")
                 audio_handler.detector.stop_interrupt_listener()
                 return ("ERROR", f"Unexpected response type: {type(response_source)}")

            # Wait for Playback Completion
            if not interrupted:
                print("Waiting for audio playback to complete...")
                wait_start_time = time.time()
                while audio_handler.player.is_playing:
                    if interrupt_event.is_set(): interrupted = True; break
                    if time.time() - wait_start_time > 60: interrupted = True; break
                    time.sleep(0.1)

            # Cleanup and Return Status
            if interrupted:
                print("Stopping playback due to interrupt.")
                audio_handler.stop_playback() # Stops player AND detector
                return ("INTERRUPTED", full_response_text)
            else:
                print("Playback completed.")
                audio_handler.detector.stop_interrupt_listener() # Stop only detector
                audio_handler.player.wait_for_playback_complete(timeout=5)
                return ("COMPLETED", full_response_text)

        except AttributeError as e:
            print(f"Error accessing TTS/Audio components: {e}. Was initialization successful?")
            traceback.print_exc()
            # Attempt cleanup even if components were problematic
            if 'audio_handler' in locals() and audio_handler: audio_handler.stop_playback()
            return ("ERROR", str(e))
        except Exception as e:
            print(f"\nError in speak method: {e}")
            traceback.print_exc()
            if 'audio_handler' in locals() and audio_handler: audio_handler.stop_playback()
            return ("ERROR", str(e)) 

    def interaction_loop(self, duration=None, timeout=10, phrase_limit=10):
        """Core interaction loop using component manager."""
        audio_handler = self.component_manager.audio_handler
        try:
            # Stop any playback before listening
            if audio_handler:
                 # Check player status directly
                 if audio_handler.player.is_playing:
                      print("Stopping playback before new interaction loop...")
                      audio_handler.stop_playback()
                      # Wait briefly for stop command to take effect
                      time.sleep(0.1) 
            else:
                 print("Warning: Audio handler not available at start of interaction loop.")
                 # Cannot proceed without audio handler
                 return "ERROR", "Audio handler not initialized."

            # Listen for user input
            print("\nListening for your voice...")
            transcribed_text = self.listen(duration=duration, timeout=timeout)

            # Handle listen errors
            if transcribed_text in ["TIMEOUT_ERROR", "low_energy", "", "ERROR", None]:
                 ai_response_text = f"Sorry, I encountered an issue: {transcribed_text}. Please try again."
                 if transcribed_text in ["TIMEOUT_ERROR", "low_energy", ""]:
                      ai_response_text = f"I didn't quite catch that ({transcribed_text or 'no input'}). Could you please repeat?"
                 print("\nAssistant:", ai_response_text)
                 return "", ai_response_text 

            # --- Process valid input ---
            print(f"\nYou: {transcribed_text}")
            self.conversation_manager.add_user_message(transcribed_text)

            # Get response from LLM
            print("\nAssistant thinking...")
            response_source = self.process_and_respond()

            # Speak the response
            speak_status, ai_response_text = self.speak(response_source)

            # Add assistant response to history
            if ai_response_text:
                self.conversation_manager.add_assistant_message(ai_response_text)

            # Handle speak status
            if speak_status == "INTERRUPTED":
                return "INTERRUPTED", ai_response_text 
            elif speak_status == "ERROR":
                 # Error message might be in ai_response_text from speak
                return "ERROR", ai_response_text 
            # DISABLED and COMPLETED cases fall through

            return transcribed_text, ai_response_text

        except Exception as e:
            print(f"\nError in interaction loop: {str(e)}")
            traceback.print_exc()
            if audio_handler:
                 audio_handler.stop_playback()
            return "ERROR", str(e)

    def main_loop(self):
        """The main loop calling interaction_loop and handling cleanup via ComponentManager."""
        print(f"Starting main loop with timeout={self._timeout_arg}, phrase_limit={self._phrase_limit_arg}, duration={self._duration_arg}")
        try:
            while True:
                user_input_status, assistant_output = self.interaction_loop(
                    duration=self._duration_arg,
                    timeout=self._timeout_arg,
                    phrase_limit=self._phrase_limit_arg
                )
                if user_input_status == "ERROR":
                    print(f"Recovering from interaction error: {assistant_output}")
                    time.sleep(2)
                elif user_input_status == "INTERRUPTED":
                     print("Interaction interrupted, starting new loop.")

        except KeyboardInterrupt:
             print("\nExiting Voice Assistant...")
        finally:
            print("Performing final cleanup via ComponentManager...")
            if hasattr(self, 'component_manager') and self.component_manager:
                 self.component_manager.cleanup()
            # Also cleanup conversation manager if needed (though less critical)
            if hasattr(self, 'conversation_manager'):
                 del self.conversation_manager
                 self.conversation_manager = None
            gc.collect()
            print("Cleanup complete.")
            