# src/core/component_manager.py

# Actual handler imports
from ..components.audio_handler import AudioHandler
from ..components.stt_handler import Transcriber
from ..components.llm_handler import LLMHandler
from ..components.tts_handler import TTSHandler
import traceback
import gc

class ComponentManager:
    def __init__(self, asr_config=None, tts_config=None, llm_config=None):
        """Initialize and hold the handler components."""
        self.asr_config = asr_config or {}
        self.tts_config = tts_config or {}
        self.llm_config = llm_config or {}

        # Handler instances
        self.audio_handler = None
        self.transcriber = None
        self.llm_handler = None
        self.tts_handler = None
        self.tts_enabled = False

        print("ComponentManager initialized.")
        self.load_all_components() # Load components on initialization

    def load_all_components(self):
        """Load all handler components sequentially."""
        print("\n--- Loading All Components --- ")
        self.load_audio_handler()
        self.load_stt()
        self.load_llm_handler()
        self.load_tts_handler()
        print("--- All Components Loaded ---\n")
        # Print summary after loading
        self._print_component_summary()
        
    def unload_component(self, component_obj, component_name):
        if component_obj:
            print(f"Unloading existing {component_name}...")
            del component_obj
            gc.collect()
            print(f"{component_name} unloaded successfully.")
            return None
        else:
            print(f"No existing {component_name} found")
            return None

    def load_audio_handler(self):
        """Load the audio handler component."""
        print("Loading Audio Handler...")
        self.audio_handler = self.unload_component(self.audio_handler, "audio_handler")
        try:
            # Pass only the ASR config as it contains relevant audio params
            self.audio_handler = AudioHandler(config=self.asr_config)
            print("Audio handler initialized successfully.")
        except Exception as e:
            print(f"Error initializing audio handler: {str(e)}")
            traceback.print_exc()
            self.audio_handler = None # Ensure it's None on failure

    def load_stt(self):
        """Load the transcription (STT) component."""
        print("Loading STT Handler (Transcriber)...")
        self.transcriber = self.unload_component(self.transcriber, "transcriber")
        try:
            model_id = self.asr_config.get('model_id', 'Unknown ASR Model')
            print(f"Initializing transcriber with config for model: {model_id}")
            # Pass the ASR config
            self.transcriber = Transcriber(config=self.asr_config)
            print("Transcriber initialized successfully.")
        except Exception as e:
            print(f"Error initializing transcriber: {str(e)}")
            traceback.print_exc()
            self.transcriber = None # Ensure it's None on failure

    def load_llm_handler(self):
        """Load the LLM handler."""
        print("Loading LLM Handler...")
        # LLM Handler might not need unloading/reloading frequently
        if self.llm_handler is None:
             try:
                  # Pass the LLM config
                  self.llm_handler = LLMHandler(config=self.llm_config)
                  print(f"LLM handler initialized successfully (Model: {getattr(self.llm_handler, 'model_name', 'N/A')}).")
             except Exception as e:
                  print(f"Error initializing LLM Handler: {str(e)}")
                  traceback.print_exc()
                  self.llm_handler = None # Ensure it's None on failure
        else:
             print("LLM Handler already loaded.")

    def load_tts_handler(self):
        """Load the TTS handler."""
        print("Loading TTS Handler...")
        self.tts_handler = self.unload_component(self.tts_handler, "tts_handler")
        self.tts_enabled = False # Assume disabled until loaded
        try:
            # Pass the TTS config
            self.tts_handler = TTSHandler(config=self.tts_config)
            self.tts_enabled = True
            print("TTS handler loaded successfully.")
        except ImportError as e:
             print(f"Error loading TTS handler: {e}. TTS might be unavailable if dependencies are missing.")
             self.tts_handler = None
        except Exception as e:
            print(f"Error loading TTS handler: {e}")
            traceback.print_exc()
            self.tts_handler = None
            
    def _print_component_summary(self):
        """Prints a summary of loaded components and their main settings."""
        print("--- Component Summary ---")
        # STT / Transcriber
        stt_model = self.asr_config.get('model', 'N/A')
        print(f"Transcription model: {stt_model}")
        device_info = "Unknown Device"
        if self.transcriber:
             if hasattr(self.transcriber, 'device'):
                 device_info = self.transcriber.device
             elif hasattr(self.transcriber, 'model') and hasattr(self.transcriber.model, 'device'): 
                  device_info = f"{self.transcriber.model.device} (compute_type: {self.transcriber.model.compute_type})"
             elif hasattr(self.transcriber, 'pipe') and hasattr(self.transcriber.pipe, 'device'):
                  device_info = str(self.transcriber.pipe.device)
        print(f"STT Device: {device_info}")
        
        # TTS
        tts_model = self.tts_config.get('model', 'N/A')
        kokoro_conf = self.tts_config.get('kokoro', {})
        tts_voice = kokoro_conf.get('voice', 'N/A')
        tts_speed = kokoro_conf.get('speed', 'N/A')
        print(f"TTS Model: {tts_model} (Enabled: {self.tts_enabled})")
        if self.tts_enabled:
            print(f"TTS Voice: {tts_voice}")
            print(f"TTS Speed: {tts_speed}x")
            
        # LLM
        llm_model = getattr(self.llm_handler, 'model_name', 'N/A')
        print(f"LLM Model: {llm_model}")
        
        print("-------------------------")

    def cleanup(self):
        """Clean up all managed components by deleting them."""
        print("Cleaning up components in ComponentManager...")
        # Delete handlers in reverse order of typical dependency/loading
        # This relies on the handlers' __del__ methods (if they exist) for actual cleanup
        if hasattr(self, 'tts_handler') and self.tts_handler:
            print("Deleting TTS handler...")
            del self.tts_handler
            self.tts_handler = None
            self.tts_enabled = False
        if hasattr(self, 'llm_handler') and self.llm_handler:
             print("Deleting LLM handler...")
             del self.llm_handler
             self.llm_handler = None
        if hasattr(self, 'transcriber') and self.transcriber:
             print("Deleting Transcriber (STT handler)...")
             del self.transcriber
             self.transcriber = None
        if hasattr(self, 'audio_handler') and self.audio_handler:
            print("Deleting Audio handler...")
            # Audio handler's __del__ should call player/detector cleanup
            del self.audio_handler
            self.audio_handler = None
            
        gc.collect() # Suggest garbage collection after deleting
        print("ComponentManager cleanup finished.") 