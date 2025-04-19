import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
rag_path = os.path.join(project_root, 'src', 'rag')
if rag_path not in sys.path:
    sys.path.insert(0, rag_path)

from src.core.alpaca import Alpaca
from src.utils.config_loader import ConfigLoader
import sys
import time # Needed for sleep in error recovery
import traceback
from dotenv import load_dotenv
from src.rag.indexer import run_indexing

def main():
    # Load environment variables from .env file first
    load_dotenv()
    print("Initializing AI Voice assistant...")
    assistant = None # Initialize assistant to None for finally block
    
    # --- Run RAG Indexing First ---
    # Execute the indexing process
    try:
        print("--- Running RAG Indexing --- ")
        run_indexing() # This will load .env internally
        print("--- RAG Indexing Complete --- \n")
    except Exception as e:
        print(f"***** CRITICAL ERROR DURING RAG INDEXING *****: {e}")
        print("***** RAG features may be unavailable or outdated. *****")
        traceback.print_exc()
        # Decide if you want to exit or continue without updated RAG
        # sys.exit(1) 
    # --- End RAG Indexing ---

    # Load configurations using ConfigLoader
    config_loader = ConfigLoader()
    assistant_params = config_loader.load_all()

    if assistant_params is None:
         print("Failed to load configurations. Exiting.")
         sys.exit(1)

    try:
         # Initialize the voice assistant core (which initializes managers and handlers)
         assistant = Alpaca(**assistant_params)

         # Extract loop parameters (consider making these instance vars of Alpaca if preferred)
         duration = assistant.duration_arg
         timeout = assistant.timeout_arg
         phrase_limit = assistant.phrase_limit_arg

         print(f"Starting main loop with timeout={timeout}, phrase_limit={phrase_limit}, duration={duration}")

         # --- Main Interaction Loop --- 
         while True:
              # Call the interaction handler's method
              user_input_status, assistant_output = assistant.interaction_handler.run_single_interaction(
                   duration=duration,
                   timeout=timeout,
                   phrase_limit=phrase_limit
              )
              
              # Handle interaction status
              if user_input_status == "ERROR":
                   print(f"Recovering from interaction error: {assistant_output}")
                   time.sleep(2)
              elif user_input_status == "INTERRUPTED":
                   print("Interaction interrupted, starting new loop.")
              # COMPLETED/DISABLED/etc. statuses just continue the loop

    except KeyboardInterrupt:
         print("\nKeyboard interrupt detected. Exiting...")
    except Exception as e:
         print(f"\nAn unexpected fatal error occurred in the main execution: {e}")
         traceback.print_exc()
    finally:
         # --- Cleanup --- 
         print("Performing final cleanup...")
         # Use ComponentManager for handler cleanup via Alpaca instance
         if assistant and hasattr(assistant, 'component_manager'):
              assistant.component_manager.cleanup()
         # Optional: Explicitly delete ConversationManager if needed
         if assistant and hasattr(assistant, 'conversation_manager'):
              del assistant.conversation_manager
          
         print("AI Voice assistant shut down.")

if __name__ == "__main__":
    main()