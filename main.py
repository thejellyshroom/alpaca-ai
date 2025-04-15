from src.core.voice_assistant import VoiceAssistant
from src.utils.config_loader import ConfigLoader
from src.rag.importdocs import importdocs
import sys

def main():
    print("Initializing AI Voice Assistant...")
    
    # Run importdocs.py first (ensure ChromaDB is populated)
    try:
        print("Running document import/check...")
        importdocs()
        print("Document import/check complete.")
    except Exception as e:
        print(f"Error during document import: {e}. RAG features might be affected.")

    # Load configurations using ConfigLoader
    config_loader = ConfigLoader()
    assistant_params = config_loader.load_all()

    if assistant_params is None:
         print("Failed to load configurations. Exiting.")
         sys.exit(1)

    # Initialize the voice assistant with loaded parameters
    # VoiceAssistant constructor now accepts all necessary params
    try:
         assistant = VoiceAssistant(**assistant_params)
    except Exception as e:
         print(f"Fatal error initializing VoiceAssistant: {e}")
         import traceback
         traceback.print_exc()
         sys.exit(1)

    # Start the main interaction loop (no longer needs args passed here)
    try:
         assistant.main_loop()
    except Exception as e:
         print(f"An unexpected error occurred in the main loop: {e}")
         import traceback
         traceback.print_exc()
         # Attempt cleanup even after main loop error
         if assistant and hasattr(assistant, 'component_manager'):
              print("Attempting cleanup after main loop error...")
              assistant.component_manager.cleanup()
    finally:
         # The main cleanup is now handled within VoiceAssistant.main_loop's finally block
         print("AI Voice Assistant shutting down.")

if __name__ == "__main__":
    main()