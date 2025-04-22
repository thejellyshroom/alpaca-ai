import time
import traceback
import types
import threading
import asyncio
import sys
from typing import Optional

from ..utils.component_manager import ComponentManager
from .tts_handler import TTSHandler
from .audio_handler import AudioHandler

class OutputHandler:
    """Handles converting LLM responses to speech and managing playback with interruptions."""
    
    def __init__(self, component_manager: ComponentManager):
        """Initializes the handler with the component manager."""
        self.component_manager = component_manager
        print("OutputHandler initialized.")
        
    def _process_tts_buffer(self, tts_buffer: str, initial_words_spoken: bool, interrupt_event: threading.Event) -> tuple[str, bool, bool]:
        """Determines if a chunk should be spoken, synthesizes/plays, returns updated buffer & state."""
        # Access components via the stored manager
        tts_handler: Optional[TTSHandler] = self.component_manager.tts_handler
        audio_handler: Optional[AudioHandler] = self.component_manager.audio_handler
        interrupted = False
        speak_this_chunk = False
        sentence_ends = (".", "!", "?", "\n", ",", ";", "–")
        approx_words_for_initial_chunk = 8
        
        if not tts_handler or not audio_handler or not audio_handler.player:
             print("Warning: TTS or Audio components not available in _process_tts_buffer.")
             # Return buffer unmodified, assume not spoken, not interrupted
             return tts_buffer, initial_words_spoken, False 

        # Determine if we should speak this chunk
        if not initial_words_spoken:
            word_count = tts_buffer.count(' ') + 1 
            if word_count >= approx_words_for_initial_chunk or any(tts_buffer.endswith(punc) for punc in sentence_ends):
                speak_this_chunk = True
        else:
            if any(tts_buffer.endswith(punc) for punc in sentence_ends):
                 speak_this_chunk = True
                 
        # Synthesize and Play if needed
        if speak_this_chunk and tts_buffer.strip():
            chunk_to_speak = tts_buffer.strip()
            remaining_buffer = "" # Reset buffer after speaking
            initial_words_spoken = True # Mark initial chunk spoken
            try:
                audio_array, sample_rate = tts_handler.synthesize(chunk_to_speak)
                # Check interrupt *before* playing
                if interrupt_event.is_set(): 
                     interrupted = True
                     print("[TTS Buffer] Interrupt detected before playing chunk.")
                # Only play if not interrupted and audio is valid
                if not interrupted and audio_array is not None and len(audio_array) > 0:
                    audio_handler.player.play_audio(audio_array, sample_rate)
                    # Check interrupt *during* playback (important for short chunks)
                    if interrupt_event.is_set(): interrupted = True; audio_handler.player.stop_playback()
                elif not interrupted:
                    print(f"[Debug TTS Buffer] Skipping play_audio for chunk due to invalid audio_array or interruption.") 
            except Exception as e:
                 print(f"\nError during TTS synthesis/playback for chunk: {e}") 
                 interrupted = True # Treat TTS error as an interruption to stop stream
        else:
            # If not speaking, keep the buffer as is
            remaining_buffer = tts_buffer

        return remaining_buffer, initial_words_spoken, interrupted
    # --- End Helper --- 

    async def speak(self, response_source):
        """Convert text response (string or generator) to speech (Moved from AlpacaInteraction._speak)."""
        tts_handler: Optional[TTSHandler] = self.component_manager.tts_handler
        audio_handler: Optional[AudioHandler] = self.component_manager.audio_handler
        # Assuming AudioHandler has the detector reference needed for interrupt listening
        detector = getattr(audio_handler, 'detector', None) if audio_handler else None
        tts_enabled = self.component_manager.tts_enabled

        if not tts_enabled or not audio_handler or not tts_handler or not detector:
            print("TTS is disabled or handlers/detector not available. Cannot speak.")
            full_response_text = ""
            # Consume generator if needed, even if not speaking
            if isinstance(response_source, (types.GeneratorType, types.AsyncGeneratorType)):
                 try: 
                      if isinstance(response_source, types.AsyncGeneratorType):
                           full_response_text = "".join([item async for item in response_source])
                      else:
                           full_response_text = "".join(list(response_source))
                 except Exception as e:
                      print(f"Error consuming response generator while TTS disabled: {e}")
                 print(f"assistant (TTS Disabled): {full_response_text}") 
            elif isinstance(response_source, str):
                 full_response_text = response_source
                 print(f"assistant (TTS Disabled): {full_response_text}")
            return ("DISABLED", full_response_text) 

        interrupt_event = threading.Event() 
        interrupted = False
        full_response_text = ""
        tts_buffer = ""
        initial_words_spoken = False
        playback_completed_normally = True # Track if playback finished vs error/timeout
        
        try:
            print("Assistant:", end="", flush=True)
            # Check if interruptions are enabled before starting listener
            if self.component_manager.interruptions_enabled:
                if hasattr(audio_handler, 'start_interrupt_listener'):
                    print("[OutputHandler] Interruptions enabled, starting listener...")
                    audio_handler.start_interrupt_listener(interrupt_event)
                else:
                    print("Warning: Interruptions enabled but AudioHandler has no start_interrupt_listener.")
                    # Fallback: create an event that never gets set
                    interrupt_event = threading.Event() 
            else:
                print("[OutputHandler] Interruptions disabled, listener not started.")
                # Ensure the event is never set if listener is disabled
                interrupt_event = threading.Event()

            # --- Handle Async Generator --- 
            if isinstance(response_source, types.AsyncGeneratorType):
                async for token in response_source: 
                    if interrupt_event.is_set(): interrupted = True; break
                    print(token, end="", flush=True) 
                    full_response_text += token
                    tts_buffer += token
                    
                    tts_buffer, initial_words_spoken, chunk_interrupted = self._process_tts_buffer(tts_buffer, initial_words_spoken, interrupt_event)
                    if chunk_interrupted: interrupted = True; break
                    # Small yield allows interrupt listener thread to run
                    await asyncio.sleep(0.01)
                print() # Newline after loop

            # --- Handle Sync Generator --- 
            elif isinstance(response_source, types.GeneratorType):
                 for token in response_source:
                     if interrupt_event.is_set(): interrupted = True; break
                     print(token, end="", flush=True) 
                     full_response_text += token
                     tts_buffer += token
                     
                     tts_buffer, initial_words_spoken, chunk_interrupted = self._process_tts_buffer(tts_buffer, initial_words_spoken, interrupt_event)
                     if chunk_interrupted: interrupted = True; break
                     # Needs yield in sync loop too for interrupt listener
                     time.sleep(0.01) 
                 print() # Newline after loop

            # --- Handle String Input --- 
            elif isinstance(response_source, str):
                print(response_source)
                full_response_text = response_source
                if full_response_text.strip():
                    try:
                        # Process the whole string using the buffer logic for consistency
                        tts_buffer, initial_words_spoken, chunk_interrupted = self._process_tts_buffer(full_response_text, False, interrupt_event)
                        if chunk_interrupted: interrupted = True
                    except Exception as e:
                        print(f"\nError synthesizing/playing full string: {e}")
                        playback_completed_normally = False
            # --- Handle Unexpected Type --- 
            else:
                 print(f"\nError: OutputHandler.speak received unexpected type: {type(response_source)}")
                 return ("ERROR", f"Unexpected response type: {type(response_source)}")

            # --- Final Buffer Handling (Common Logic) --- 
            if not interrupted and tts_buffer.strip():
                 print(f"\n[OutputHandler] Processing final buffer: '{tts_buffer[:50]}...'")
                 try:
                     _, _, chunk_interrupted = self._process_tts_buffer(tts_buffer.strip(), initial_words_spoken, interrupt_event)
                     if chunk_interrupted: interrupted = True
                 except Exception as e: 
                      print(f"\nError synthesizing/playing final segment: {e}")
                      playback_completed_normally = False
            # --- End Final Buffer Handling --- 
            
            # --- Wait for Playback Completion (Common Logic) ---
            if not interrupted:
                wait_start_time = time.time()
                playback_timeout = 60 # seconds
                while audio_handler.player.is_playing:
                    if interrupt_event.is_set(): interrupted = True; break
                    if time.time() - wait_start_time > playback_timeout: 
                         print(f"\nTTS Playback Wait Timeout ({playback_timeout}s)")
                         playback_completed_normally = False 
                         break 
                    # Use asyncio sleep for async context, time.sleep if called synchronously (though speak is async)
                    await asyncio.sleep(0.1)

            # --- Return Status (Common Logic) --- 
            if interrupted:
                print("\n[OutputHandler] Stopping playback due to interrupt.")
                # Stop playback via audio_handler
                if hasattr(audio_handler, 'stop_playback'): audio_handler.stop_playback()
                return ("INTERRUPTED", full_response_text)
            elif playback_completed_normally:
                 print("\n[OutputHandler] Playback completed.")
                 return ("COMPLETED", full_response_text)
            else:
                 print("\n[OutputHandler] Playback did not complete normally (error/timeout).")
                 if hasattr(audio_handler, 'stop_playback'): audio_handler.stop_playback()
                 return ("ERROR", full_response_text)

        except Exception as e:
            print(f"\nError in OutputHandler.speak method: {e}")
            traceback.print_exc()
            if hasattr(audio_handler, 'stop_playback'): audio_handler.stop_playback()
            return ("ERROR", str(e)) 
        finally:
            # Stop the interrupt listener
            if hasattr(audio_handler, 'stop_interrupt_listener'):
                 try:
                      audio_handler.stop_interrupt_listener()
                      print("[OutputHandler] Interrupt listener stopped.")
                 except Exception as e_stop:
                      print(f"Error stopping interrupt listener: {e_stop}")
            else:
                 print("Warning: AudioHandler has no stop_interrupt_listener method.") 