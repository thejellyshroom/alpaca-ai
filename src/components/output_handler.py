import time
import traceback
import types
import threading
import asyncio
from asyncio import Queue
import sys
from typing import Optional
import base64
import numpy as np

from utils.component_manager import ComponentManager
from components.tts_handler import TTSHandler
from components.audio_handler import AudioHandler

class OutputHandler:
    """Handles converting LLM responses to speech and managing playback with interruptions."""
    
    def __init__(self, component_manager: ComponentManager):
        """Initializes the handler with the component manager."""
        self.component_manager = component_manager
        print("OutputHandler initialized.")
        
    async def _process_tts_buffer(self, tts_buffer: str, initial_words_spoken: bool, interrupt_event: threading.Event, status_queue: Optional[Queue]) -> tuple[str, bool, bool]:
        """Determines if a chunk should be spoken, synthesizes, sends audio via queue, returns updated buffer & state."""
        tts_handler: Optional[TTSHandler] = self.component_manager.tts_handler
        audio_handler: Optional[AudioHandler] = self.component_manager.audio_handler
        interrupted = False
        speak_this_chunk = False
        sentence_ends = (".", "!", "?", "\n", ",", ";", "–")
        approx_words_for_initial_chunk = 8
        
        if not tts_handler:
             print("Warning: TTS handler not available in _process_tts_buffer.")
             return tts_buffer, initial_words_spoken, False 

        if not initial_words_spoken:
            word_count = tts_buffer.count(' ') + 1 
            if word_count >= approx_words_for_initial_chunk or any(tts_buffer.endswith(punc) for punc in sentence_ends):
                speak_this_chunk = True
        else:
            if any(tts_buffer.endswith(punc) for punc in sentence_ends):
                 speak_this_chunk = True
                 
        if speak_this_chunk and tts_buffer.strip():
            chunk_to_speak = tts_buffer.strip()
            remaining_buffer = "" 
            initial_words_spoken = True 
            try:
                print(f"    \n---> Synthesizing chunk: '{chunk_to_speak}'") # Log input chunk
                audio_array, sample_rate = tts_handler.synthesize(chunk_to_speak)
                
                # Log raw output type and shape/stats if numpy array
                if isinstance(audio_array, np.ndarray):
                    print(f"    <--- TTS returned numpy array | dtype: {audio_array.dtype} | shape: {audio_array.shape} | min: {np.min(audio_array):.2f} | max: {np.max(audio_array):.2f} | mean: {np.mean(audio_array):.2f}")
                elif isinstance(audio_array, bytes):
                    print(f"    <--- TTS returned bytes | len: {len(audio_array)}")
                else:
                    print(f"    <--- TTS returned unexpected type: {type(audio_array)}")

                if interrupt_event.is_set(): 
                     interrupted = True
                if not interrupted and audio_array is not None and len(audio_array) > 0:
                    if status_queue:
                        # Ensure audio_array is bytes (e.g., from numpy array)
                        # This might need adjustment based on synthesize() output type
                        audio_bytes = b''
                        if isinstance(audio_array, np.ndarray):
                            # --- Explicitly convert to int16 before sending ---
                            if np.issubdtype(audio_array.dtype, np.floating):
                                print(f"    (OutputHandler: Converting float audio to int16)")
                                # Scale float from [-1.0, 1.0] to int16 range [-32768, 32767]
                                audio_array = np.clip(audio_array, -1.0, 1.0)
                                audio_array = (audio_array * 32767).astype(np.int16)
                            elif audio_array.dtype != np.int16:
                                print(f"    (OutputHandler: Warning - Unexpected numpy dtype {audio_array.dtype}, attempting astype(int16))")
                                try:
                                    audio_array = audio_array.astype(np.int16)
                                except Exception as type_e:
                                    print(f"    (OutputHandler: Failed to convert {audio_array.dtype} to int16: {type_e})")
                                    audio_array = None # Prevent sending bad data
                            
                            if audio_array is not None:
                                audio_bytes = audio_array.tobytes()
                            # --- End conversion ---
                        elif isinstance(audio_array, bytes):
                            # If already bytes, assume it's correct format (less safe)
                            print("    (OutputHandler: Received bytes directly from TTS, assuming int16)")
                            audio_bytes = audio_array

                        if audio_bytes:
                            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                            await status_queue.put({
                                "type": "audio_chunk", 
                                "data": base64_audio,
                                "sample_rate": sample_rate,
                                "format": "pcm_s16le"
                            })
            except Exception as e:
                 print(f"\nError during TTS synthesis/queue send for chunk: {e}") 
                 traceback.print_exc()
                 interrupted = True
        else:
            remaining_buffer = tts_buffer

        return remaining_buffer, initial_words_spoken, interrupted
    # --- End Helper --- 

    async def speak(self, response_source, status_queue: Optional[Queue] = None):
        """Convert text response to speech, sending status/audio via queue."""
        tts_handler: Optional[TTSHandler] = self.component_manager.tts_handler
        audio_handler: Optional[AudioHandler] = self.component_manager.audio_handler
        detector = getattr(audio_handler, 'detector', None) if audio_handler else None
        tts_enabled = self.component_manager.tts_enabled

        async def put_status(state: str, message: str = None, **kwargs):
            if status_queue:
                payload = {"type": "status", "state": state}
                if message: payload["message"] = message
                payload.update(kwargs)
                await status_queue.put(payload)
            else:
                print(f"[Output Status] {state} {f'({message})' if message else ''}")

        if not tts_enabled:
            print("TTS is disabled. Cannot speak.")
            full_response_text = ""
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
            await put_status("Disabled", "TTS is disabled on server.")
            return ("DISABLED", full_response_text) 

        if not audio_handler or not tts_handler or not detector:
             print("Audio/TTS handlers or detector not available. Cannot speak.")
             await put_status("Error", "Audio/TTS components unavailable.")
             full_response_text = ""
             if isinstance(response_source, (types.GeneratorType, types.AsyncGeneratorType)):
                if isinstance(response_source, types.AsyncGeneratorType): full_response_text = "".join([item async for item in response_source])
                else: full_response_text = "".join(list(response_source))
             elif isinstance(response_source, str): full_response_text = response_source
             return ("ERROR", full_response_text)

        interrupt_event = threading.Event() 
        interrupted = False
        full_response_text = ""
        tts_buffer = ""
        initial_words_spoken = False
        final_status_code = "ERROR"
        
        try:
            await put_status("Speaking")
            
            if self.component_manager.interruptions_enabled:
                if hasattr(audio_handler, 'start_interrupt_listener'):
                    audio_handler.start_interrupt_listener(interrupt_event)
                else:
                    print("Warning: Interruptions enabled but no start_interrupt_listener.")
                    interrupt_event = threading.Event()
            else:
                interrupt_event = threading.Event()

            # --- Handle Async Generator --- 
            if isinstance(response_source, types.AsyncGeneratorType):
                async for token in response_source: 
                    if interrupt_event.is_set(): interrupted = True; break
                    # print(token, end="", flush=True) # Replaced by queue
                    full_response_text += token
                    tts_buffer += token
                    
                    tts_buffer, initial_words_spoken, chunk_interrupted = await self._process_tts_buffer(tts_buffer, initial_words_spoken, interrupt_event, status_queue)
                    if chunk_interrupted: interrupted = True; break
                    await asyncio.sleep(0.01)
                # print() # No console print

            # --- Handle Sync Generator --- 
            elif isinstance(response_source, types.GeneratorType):
                 # Iterate directly in the async method. This is slightly blocking per token
                 # if the generator itself blocks, but allows awaiting async functions inside.
                 for token in response_source:
                     if interrupt_event.is_set(): interrupted = True; break
                     full_response_text += token
                     tts_buffer += token
                     # Await the async helper method
                     tts_buffer, initial_words_spoken, chunk_interrupted = await self._process_tts_buffer(tts_buffer, initial_words_spoken, interrupt_event, status_queue)
                     if chunk_interrupted: interrupted = True; break
                     # Yield control to the event loop
                     await asyncio.sleep(0.01) 
                 # print() # No console print

            # --- Handle String Input --- 
            elif isinstance(response_source, str):
                full_response_text = response_source
                if full_response_text.strip():
                    try:
                        tts_buffer, initial_words_spoken, chunk_interrupted = await self._process_tts_buffer(full_response_text, False, interrupt_event, status_queue)
                        if chunk_interrupted: interrupted = True
                    except Exception as e:
                        print(f"\nError synthesizing/queueing full string: {e}")
                        await put_status("Error", f"TTS Error: {e}")
                        final_status_code = "ERROR"
            else:
                 print(f"\nError: OutputHandler.speak received unexpected type: {type(response_source)}")
                 await put_status("Error", f"Unexpected response type: {type(response_source)}")
                 return ("ERROR", f"Unexpected response type: {type(response_source)}")

            if not interrupted and tts_buffer.strip():
                 try:
                     _, _, chunk_interrupted = await self._process_tts_buffer(tts_buffer.strip(), initial_words_spoken, interrupt_event, status_queue)
                     if chunk_interrupted: interrupted = True
                 except Exception as e: 
                      print(f"\nError synthesizing/queueing final segment: {e}")
                      await put_status("Error", f"TTS Error on final segment: {e}")
                      final_status_code = "ERROR"

            if interrupted:
                print("\n[OutputHandler] Interrupted during generation/TTS.")
                final_status_code = "INTERRUPTED"
                await put_status("Interrupted", "Interrupted by user/VAD.")
            elif final_status_code != "ERROR":
                 print("\n[OutputHandler] Finished generating response and sending audio chunks.")
                 final_status_code = "COMPLETED"
                 await put_status("Idle")
            else:
                 print("\n[OutputHandler] Finished generating response but encountered TTS errors.")
                 await put_status("Idle", "Finished with TTS errors.")
            
            return (final_status_code, full_response_text)

        except asyncio.CancelledError:
            print("\n[OutputHandler] Speak task cancelled.")
            raise
        except Exception as e:
            print(f"\nError in OutputHandler.speak method: {e}")
            traceback.print_exc()
            await put_status("Error", f"OutputHandler Error: {e}")
            return ("ERROR", str(e)) 
        finally:
            if hasattr(audio_handler, 'stop_interrupt_listener'):
                 try:
                      audio_handler.stop_interrupt_listener()
                 except Exception as e_stop:
                      print(f"Error stopping interrupt listener: {e_stop}")
            print("[OutputHandler] Speak method finished.") 