from ..utils.audio_handler import AudioHandler
from ..utils.transcriber import Transcriber
from ..utils.llm_handler import LLMHandler
from ..utils.tts_handler import TTSHandler
from ..utils.helper_functions import *
import os
import torch
import time
import traceback

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
            text (str): Input text to process
            
        Returns:
            str: LLM's response
        """
        query = text[-1]
        # Use RAG for knowledge-based queries
        if should_use_rag(query["content"]):
            response = self.llm_handler.get_rag_response(query["content"], self.conversation_history[:-1])
        else:
            response = self.llm_handler.get_response(self.conversation_history)
        return response
    
    def speak(self, text):
        """Convert text to speech and play it.
        
        Args:
            text (str): Text to speak
            
        Returns:
            bool: True if speech was successfully played
        """
        if not self.tts_enabled:
            print("TTS is disabled. Cannot speak the response.")
            return False
            
        try:
            # Ensure text is not None
            if text is None:
                print("Warning: Received None text to speak")
                return False

            # Split text into sentences for more natural pauses
            sentences = split_into_sentences(text)
            
            for sentence in sentences:
                # Synthesize and play each sentence
                audio_array, sample_rate = self.tts_handler.synthesize(sentence)
                    
                if len(audio_array) > 0:
                    self.audio_handler.play_audio(audio_array, sample_rate)
                    # Small pause between sentences
                    time.sleep(0.3)
                else:
                    print("Generated audio is empty. Cannot play.")
            
            # Wait for all audio to finish playing
            if self.audio_handler and self.audio_handler.is_playing:
                self.audio_handler.wait_for_playback_complete(timeout=None)
            
            return True
            
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            traceback.print_exc()
            return False
    
    def interaction_loop(self, duration=None, timeout=10, phrase_limit=10):
        """Record audio, transcribe, process with streaming response.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds
            timeout (int, optional): Maximum seconds to wait before giving up
            phrase_limit (int, optional): Maximum seconds for a single phrase
            
        Returns:
            tuple: (transcribed_text, ai_response)
        """
        try:
            # Ensure all audio playback is stopped
            if self.audio_handler:
                self.audio_handler.stop_playback()
            
            # Listen for user input
            print("\nListening for your voice...")
            transcribed_text = self.listen(duration=duration, timeout=timeout)
            
            # Handle no speech detected
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                ai_response = "Seems like you didn't say anything."
                print("\nAssistant:", ai_response)
                return "", ai_response
            
            if transcribed_text == "TIMEOUT_ERROR":
                ai_response = "Time out error occurred"
                print("\nAssistant:", ai_response)
                return transcribed_text, ai_response
            
            # Display transcribed text    
            print("\nYou said:", transcribed_text)
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": transcribed_text})
            
            try:
                # Get streaming response and process it
                response_stream = self.process_and_respond(self.conversation_history)
                assistant_buffer = process_streaming_response(
                    self.tts_handler, 
                    self.audio_handler, 
                    response_stream, 
                    self.tts_enabled
                )
            except Exception as e:
                print(f"\nError during response generation: {e}")
                assistant_buffer = "I'm sorry, I encountered an error while generating a response."
                print("\nFallback response:", assistant_buffer)
            
            # Wait for all audio to complete before returning
            if self.audio_handler and self.tts_enabled:
                self.audio_handler.wait_for_playback_complete(timeout=None)
            
            return transcribed_text, assistant_buffer
            
        except Exception as e:
            print(f"Unexpected error in streaming interaction: {str(e)}")
            traceback.print_exc()
            return "", "I'm sorry, I encountered an unexpected error."
            