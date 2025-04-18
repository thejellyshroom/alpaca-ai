# src/core/interaction_handler.py

import time
import traceback
import types
import threading

# Assuming managers and handlers are now in sibling directories or utils
# Adjust imports based on your final structure
from ..utils.component_manager import ComponentManager       # Corrected path assumed
from ..utils.conversation_manager import ConversationManager # Corrected path assumed

class AlpacaInteraction:
    def __init__(self, component_manager: ComponentManager, conversation_manager: ConversationManager):
        """Initializes the handler with necessary managers."""
        if not component_manager or not conversation_manager:
             raise ValueError("ComponentManager and ConversationManager are required.")
        self.component_manager = component_manager
        self.conversation_manager = conversation_manager
        print("AlpacaInteraction initialized.")

    def _listen(self, duration=None, timeout=None):
        """Record audio and transcribe it using ComponentManager handlers."""
        audio_handler = self.component_manager.audio_handler
        transcriber = self.component_manager.transcriber
        
        if not audio_handler or not transcriber:
             print("Error: Audio Handler or Transcriber not available in AlpacaInteraction.")
             return "ERROR"

        try:
            # Ensure any ongoing playback is stopped
            if audio_handler.player.is_playing: # Check player directly
                print("Stopping any ongoing audio playback before listening...")
                audio_handler.stop_playback()
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
             print(f"Error accessing audio/transcriber components via ComponentManager: {e}.")
             traceback.print_exc()
             return "ERROR"
        except Exception as e:
            print(f"Unexpected error in _listen method: {str(e)}")
            traceback.print_exc()
            return "ERROR"

    def _process_and_respond(self):
        """Processes the current conversation and decides whether to use RAG."""
        llm_handler = self.component_manager.llm_handler
        if not llm_handler:
             print("Error: LLM Handler not available.")
             def error_gen(): yield "Error: LLM Handler not available."; return
             return error_gen()
        
        conversation_history = self.conversation_manager.get_history()
        last_user_message = conversation_history[-1]['content'] if conversation_history and conversation_history[-1]['role'] == 'user' else ""

        # Check if RAG is available and configured
        if llm_handler.rag_collection:
            print("RAG connection available. Using get_rag_response.")
            return llm_handler.get_rag_response(query=last_user_message, messages=conversation_history)
        else:
            print("RAG connection not available. Using standard get_response.")
            return llm_handler.get_response(messages=conversation_history)

    def _speak(self, response_source):
        """Convert text to speech using ComponentManager handlers."""
        tts_handler = self.component_manager.tts_handler
        audio_handler = self.component_manager.audio_handler
        tts_enabled = self.component_manager.tts_enabled

        if not tts_enabled or not audio_handler or not tts_handler:
            print("TTS is disabled or handlers not available. Cannot speak.")
            full_response_text = ""
            if isinstance(response_source, str):
                 full_response_text = response_source
            elif isinstance(response_source, types.GeneratorType):
                 try: full_response_text = "".join(list(response_source))
                 except Exception: pass # Ignore errors consuming generator here
                 print(f"assistant (TTS Disabled): {full_response_text}") # Print consumed text
            return ("DISABLED", full_response_text) 

        interrupt_event = threading.Event()
        interrupted = False
        full_response_text = ""
        tts_buffer = ""
        sentence_ends = (".", "!", "?", "\n", ",", ";", "–")
        initial_words_spoken = False
        word_count = 0
        approx_words_for_initial_chunk = 8

        try:
            print("Assistant:", end="", flush=True)
            audio_handler.detector.start_interrupt_listener(interrupt_event)

            if isinstance(response_source, types.GeneratorType):
                for token in response_source:
                    if interrupt_event.is_set(): interrupted = True; break
                    print(token, end="", flush=True) 
                    full_response_text += token
                    tts_buffer += token
                    
                    speak_this_chunk = False
                    # --- Initial Chunk Logic --- 
                    if not initial_words_spoken:
                        word_count = tts_buffer.count(' ') + 1 
                        # Speak if enough words OR any punctuation detected
                        if word_count >= approx_words_for_initial_chunk or any(tts_buffer.endswith(punc) for punc in sentence_ends):
                            speak_this_chunk = True
                            initial_words_spoken = True
                    # --- Subsequent Chunk Logic --- 
                    else:
                        # Speak only when punctuation is detected (removed length check)
                        if any(tts_buffer.endswith(punc) for punc in sentence_ends):
                             speak_this_chunk = True
                             
                    # --- Synthesize and Play --- 
                    if speak_this_chunk and tts_buffer.strip():
                        chunk_to_speak = tts_buffer.strip()
                        tts_buffer = ""
                        word_count = 0
                        try:
                            audio_array, sample_rate = tts_handler.synthesize(chunk_to_speak)
                            if interrupt_event.is_set(): interrupted = True; break
                            if audio_array is not None and len(audio_array) > 0:
                                audio_handler.player.play_audio(audio_array, sample_rate)
                                time.sleep(0.05)
                        except Exception as e:
                             print(f"\nError during TTS synthesis for chunk: {e}")
                             time.sleep(0.1)
                print()
                if not interrupted and tts_buffer.strip():
                     try:
                         audio_array, sample_rate = tts_handler.synthesize(tts_buffer.strip())
                         if audio_array is not None and len(audio_array) > 0:
                             audio_handler.player.play_audio(audio_array, sample_rate)
                     except Exception as e: print(f"\nError synthesizing final segment: {e}")
            else:
                 print(f"\nError: _speak received unexpected type: {type(response_source)}")
                 audio_handler.detector.stop_interrupt_listener()
                 return ("ERROR", f"Unexpected response type: {type(response_source)}")

            if not interrupted:
                wait_start_time = time.time()
                while audio_handler.player.is_playing:
                    if interrupt_event.is_set(): interrupted = True; break
                    if time.time() - wait_start_time > 60: interrupted = True; print("\nTTS Wait Timeout"); break # Timeout
                    time.sleep(0.1)

            if interrupted:
                print("\nStopping playback due to interrupt.")
                audio_handler.stop_playback()
                return ("INTERRUPTED", full_response_text)
            else:
                print("\nPlayback completed.")
                audio_handler.detector.stop_interrupt_listener()
                # audio_handler.player.wait_for_playback_complete(timeout=5) # Less critical now?
                return ("COMPLETED", full_response_text)

        except AttributeError as e:
            print(f"\nError accessing TTS/Audio components in _speak: {e}.")
            traceback.print_exc()
            if 'audio_handler' in locals() and audio_handler: audio_handler.stop_playback()
            return ("ERROR", str(e))
        except Exception as e:
            print(f"\nError in _speak method: {e}")
            traceback.print_exc()
            if 'audio_handler' in locals() and audio_handler: audio_handler.stop_playback()
            return ("ERROR", str(e)) 

    # Renamed from Alpaca.interaction_loop
    def run_single_interaction(self, duration=None, timeout=10, phrase_limit=10):
        """Runs a single listen -> process -> speak cycle."""
        audio_handler = self.component_manager.audio_handler
        try:
            # Stop any playback before listening (moved from interaction_loop start)
            if audio_handler and audio_handler.player.is_playing:
                 print("Stopping playback before new interaction...")
                 audio_handler.stop_playback()
                 time.sleep(0.1) # Allow stop to propagate
            elif not audio_handler:
                 print("Error: Audio handler not available for interaction.")
                 return "ERROR", "Audio handler not initialized."

            # 1. Listen
            print("\nListening for your voice...")
            transcribed_text = self._listen(duration=duration, timeout=timeout)
            
            # Handle listen errors
            if transcribed_text in ["TIMEOUT_ERROR", "low_energy", "", "ERROR", None]:
                 ai_response_text = f"Sorry, I encountered an issue: {transcribed_text}. Please try again."
                 if transcribed_text in ["TIMEOUT_ERROR", "low_energy", ""]:
                      ai_response_text = f"I didn't quite catch that ({transcribed_text or 'no input'}). Could you please repeat?"
                 print("\nassistant:", ai_response_text)
                 # Return the error status, and the generated message
                 return (transcribed_text or "ERROR"), ai_response_text 

            # --- Process valid input ---
            print(f"\nYou: {transcribed_text}")
            # 2. Add user message to history
            self.conversation_manager.add_user_message(transcribed_text)

            # 3. Process and get response
            print("\nassistant thinking...")
            response_source = self._process_and_respond()

            # 4. Speak
            speak_status, ai_response_text = self._speak(response_source)

            # 5. Add assistant message to history
            if ai_response_text:
                 self.conversation_manager.add_assistant_message(ai_response_text)

            # Handle speak status - INTERRUPTED, ERROR, DISABLED, COMPLETED
            if speak_status in ["INTERRUPTED", "ERROR"]:
                 print(f"(Interaction ended with status: {speak_status})")
            # Return the status from speak, and the full text spoken/generated
            return speak_status, ai_response_text

        except Exception as e:
            print(f"\nCritical error in interaction handler: {e}")
            traceback.print_exc()
            if audio_handler: audio_handler.stop_playback()
            return "ERROR", str(e) 