# src/core/interaction_handler.py

import time
import traceback
import types
import threading
import asyncio # Add asyncio import
import soundfile as sf 
import sys # Ensure sys is imported if used elsewhere

from ..utils.component_manager import ComponentManager       # Corrected path assumed
from ..utils.conversation_manager import ConversationManager # Corrected path assumed
from ..components.output_handler import OutputHandler # Import the new handler

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
            
            # If listening was successful (got a filepath), now transcribe
            print(f"Audio saved to: {audio_result}. Transcribing...")
            
            # --- Call transcribe with the file path directly --- 
            try:
                 # Pass the file path string to the transcribe method
                 transcribed_text = transcriber.transcribe(audio_result)
                 
            except Exception as transcribe_e:
                 print(f"Error during transcription: {transcribe_e}")
                 traceback.print_exc()
                 return "ERROR" # Error during transcription phase
            # --- End file path transcription ---

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

    async def _process_and_respond(self):
        """Processes the current conversation and decides whether to use RAG asynchronously."""
        llm_handler = self.component_manager.llm_handler
        if not llm_handler:
             print("Error: LLM Handler not available.")
             async def error_gen(): 
                 yield "Error: LLM Handler not available."
             return error_gen()
        
        conversation_history = self.conversation_manager.get_history()
        last_user_message = conversation_history[-1]['content'] if conversation_history and conversation_history[-1]['role'] == 'user' else ""

        if llm_handler.rag_querier: # Check for the initialized MiniRAG querier instance
            print("RAG querier available.")
            return await llm_handler.get_rag_response(query=last_user_message, messages=conversation_history)
        else:
            print("RAG not available or disabled.")
            return llm_handler.get_response(messages=conversation_history) 

    async def run_single_interaction(self, duration=None, timeout=10, phrase_limit=10):
        """Runs a single listen -> process -> speak cycle asynchronously."""
        audio_handler = self.component_manager.audio_handler
        try:
            if audio_handler and audio_handler.player.is_playing:
                 print("Stopping playback before new interaction...")
                 audio_handler.stop_playback()
                 time.sleep(0.1) # Allow stop to propagate
            elif not audio_handler:
                 print("Error: Audio handler not available for interaction.")
                 return "ERROR", "Audio handler not initialized."

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
            self.conversation_manager.add_user_message(transcribed_text)
            print("\nAlpaca is thinking...")
            response_source = await self._process_and_respond() # Use await here
            speak_status, ai_response_text = await self.output_handler.speak(response_source)
            if ai_response_text:
                 self.conversation_manager.add_assistant_message(ai_response_text)
            if speak_status in ["INTERRUPTED", "ERROR"]:
                 print(f"(Interaction ended with status: {speak_status})")
            return speak_status, ai_response_text

        except Exception as e:
            print(f"\nCritical error in interaction handler: {e}")
            traceback.print_exc()
            if audio_handler: audio_handler.stop_playback()
            return "ERROR", str(e) 