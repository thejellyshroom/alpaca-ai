# src/core/interaction_handler.py

import time
import traceback
import types
import threading
import asyncio
from asyncio import Queue # Import Queue
import soundfile as sf 
import sys # Ensure sys is imported if used elsewhere

from utils.component_manager import ComponentManager
from utils.conversation_manager import ConversationManager
from components.output_handler import OutputHandler

class AlpacaInteraction:
    def __init__(self, component_manager: ComponentManager, conversation_manager: ConversationManager):
        """Initializes the handler with necessary managers and creates OutputHandler."""
        if not component_manager or not conversation_manager:
             raise ValueError("ComponentManager and ConversationManager are required.")
        self.component_manager = component_manager
        self.conversation_manager = conversation_manager
        self.output_handler = OutputHandler(self.component_manager)
        print("AlpacaInteraction initialized.")

    def _listen(self, duration=None, timeout=None):
        """Uses AudioHandler to listen for speech and Transcriber to convert it to text."""
        audio_handler = self.component_manager.audio_handler
        transcriber = self.component_manager.transcriber
        
        if not audio_handler or not transcriber:
             print("Error: Audio Handler or Transcriber not available.")
             return "ERROR"

        try:
            print("Starting new listening session...")
            # Call listen_for_speech - it returns the file path or an error code
            audio_result = audio_handler.listen_for_speech(
                filename="prompt.wav", # Keep saving for debug
                timeout=timeout,
                stop_playback=True # Ensure playback stops
            )
                
            # Handle errors from listen_for_speech
            if audio_result in ["low_energy", "TIMEOUT_ERROR", None]:
                print(f"Listening failed or timed out: {audio_result}")
                return audio_result or "ERROR" # Return code or general error
            
            print(f"Audio saved to: {audio_result}. Transcribing...")
            
            try:
                 transcribed_text = transcriber.transcribe(audio_result)
                 
            except Exception as transcribe_e:
                 print(f"Error during transcription: {transcribe_e}")
                 traceback.print_exc()
                 return "ERROR" # Error during transcription phase

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
        """Processes the current conversation and decides whether to use RAG synchronously."""
        llm_handler = self.component_manager.llm_handler
        if not llm_handler:
             print("Error: LLM Handler not available.")
             return "ERROR: LLM Handler not available." # Or raise an exception
        
        conversation_history = self.conversation_manager.get_history()
        last_user_message = conversation_history[-1]['content'] if conversation_history and conversation_history[-1]['role'] == 'user' else ""

        if llm_handler.rag_querier: # Check for the initialized MiniRAG querier instance
            print("RAG querier available.")
            return llm_handler.get_rag_response(query=last_user_message, messages=conversation_history)
        else:
            print("RAG not available or disabled.")
            return llm_handler.get_response(messages=conversation_history) 

    async def run_single_interaction(self, 
                                     duration=None, 
                                     timeout=10, 
                                     phrase_limit=10,
                                     status_queue: asyncio.Queue = None): # Add queue argument
        """Runs a single listen -> process -> speak cycle asynchronously, reporting status via queue."""
        audio_handler = self.component_manager.audio_handler
        
        async def put_status(state: str, message: str = None, **kwargs):
            """Helper to put status updates onto the queue if it exists."""
            if status_queue:
                payload = {"type": "status", "state": state}
                if message: payload["message"] = message
                payload.update(kwargs)
                await status_queue.put(payload)
            else:
                # Fallback print if no queue (e.g., called directly without API)
                print(f"[Interaction Status] {state} {f'({message})' if message else ''}")

        try:
            await put_status("InitializingInteraction")
            if audio_handler and audio_handler.player.is_playing:
                 print("Stopping playback before new interaction...")
                 # TODO: Maybe report this via queue?
                 audio_handler.stop_playback()
                 await asyncio.sleep(0.1) # Allow stop to propagate (use asyncio.sleep)
            elif not audio_handler:
                 print("Error: Audio handler not available for interaction.")
                 await put_status("Error", "Audio handler not initialized.")
                 return "ERROR", "Audio handler not initialized."

            # --- Listening Phase ---
            await put_status("Listening")
            print("[Interaction] Running _listen in executor...")
            transcribed_text = await asyncio.to_thread(
                self._listen, duration=duration, timeout=timeout
            )
            print(f"[Interaction] _listen result: '{transcribed_text[:50]}...'")
            
            # Report transcription result or error via queue
            if status_queue:
                if transcribed_text not in ["TIMEOUT_ERROR", "low_energy", "", "ERROR", None]:
                    await status_queue.put({"type": "transcript", "text": transcribed_text})
                else:
                    # Send error status if listening failed
                    await put_status("Error", f"Listening failed: {transcribed_text or 'Unknown Error'}")
            
            # Handle listen errors (copied logic, but status already sent)
            if transcribed_text in ["TIMEOUT_ERROR", "low_energy", "", "ERROR", None]:
                 # The AI response text here is mainly for the internal conversation history
                 # The actual error state was already sent via the queue
                 ai_response_text = f"Sorry, I encountered an issue: {transcribed_text}. Please try again."
                 if transcribed_text in ["TIMEOUT_ERROR", "low_energy", ""]:
                      ai_response_text = f"I didn't quite catch that ({transcribed_text or 'no input'}). Could you please repeat?"
                 print(f"(Internal handling for listen error: {ai_response_text})")
                 # Return error status and internal message
                 return (transcribed_text or "ERROR"), ai_response_text 
            # --- End Listening Phase ---

            # --- Processing Phase ---
            # print(f"\nYou: {transcribed_text}") # Already sent via queue
            self.conversation_manager.add_user_message(transcribed_text)
            await put_status("Processing")
            # print("\nAlpaca is thinking...") # Status sent via queue
            response_source = await self._process_and_respond() # Returns generator
            # --- End Processing Phase ---

            # --- Speaking Phase (OutputHandler needs modification too) ---
            await put_status("SpeakingInitialization") # Indicate TTS might start soon
            # OutputHandler.speak needs to accept the queue to send audio chunks and final status
            speak_status, ai_response_text = await self.output_handler.speak(response_source, status_queue=status_queue)
            
            if ai_response_text: # Add the final concatenated text from TTS to history
                 self.conversation_manager.add_assistant_message(ai_response_text)
            
            # Final status (Idle, Interrupted, Error) should be sent by output_handler via the queue
            # We just return the final status and text from speak()
            print(f"(Interaction cycle ended with speak_status: {speak_status})")
            return speak_status, ai_response_text
            # --- End Speaking Phase ---

        except asyncio.CancelledError:
             print("[Interaction Handler] Task cancelled.")
             await put_status("Cancelled", "Interaction task was cancelled.")
             if audio_handler: audio_handler.stop_playback()
             # Re-raise the cancellation to be caught by the caller (websocket handler)
             raise 
        except Exception as e:
            print(f"\nCritical error in interaction handler: {e}")
            traceback.print_exc()
            await put_status("Error", f"Critical error: {e}")
            if audio_handler: audio_handler.stop_playback()
            return "ERROR", str(e) 

    async def run_single_text_interaction(self, user_text: str):
        """Processes text input and returns an async generator for the response stream."""
        try:
            if not user_text:
                print("Warning: Received empty text input.")
                def empty_gen():
                    if False: yield
                return empty_gen() # Or return ""

            self.conversation_manager.add_user_message(user_text)
            response_source = self._process_and_respond() 
            return response_source

        except Exception as e:
            print(f"\nCritical error in text interaction handler: {e}")
            traceback.print_exc()
            error_message = str(e)
            def error_gen():
                yield f"[Critical Error: {error_message}]" # Use the captured message
            return error_gen()