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

# --- Add imports for session summarization --- 
import asyncio
from src.utils.session_utils import summarize_conversation, save_summary
# Assume Alpaca class exposes conversation_manager and llm_handler
# If not, we might need to adjust how exit_handler gets them
# ---------------------------------------------

# --- Helper function to run async from sync --- 
def run_async_in_sync_context(async_func_coro):
    """Helper to run async function coroutine from sync context like atexit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # This is a simplified approach. Robust handling in complex async 
            # apps might require more sophisticated loop management.
            print("[Exit Handler Warning] Event loop is running. Attempting threadsafe execution.")
            future = asyncio.run_coroutine_threadsafe(async_func_coro, loop)
            return future.result() # Wait for result - blocking!
        else:
             # If no loop running, can likely use asyncio.run
            return asyncio.run(async_func_coro)
    except RuntimeError: # Handle cases where no event loop is available/set
         print("[Exit Handler] No running event loop, creating new one for summarization.")
         return asyncio.run(async_func_coro)
    except Exception as e:
         print(f"[Exit Handler] Error running async function in sync context: {e}")
         traceback.print_exc()
         return "[Error during async execution]"
# ---------------------------------------------------------

# Change main to async def
async def main():
    # Load environment variables from .env file first
    load_dotenv()
    print("Initializing AI Voice assistant...")
    assistant = None # Initialize assistant to None for finally block
    
    # --- Get DATA_PATH early --- 
    data_path_value = os.getenv("DATA_PATH", "./data/dataset") # Get path, provide default
    print(f"Using DATA_PATH: {data_path_value}")
    # ---------------------------
    
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
         # Initialize the voice assistant core
         assistant = Alpaca(**assistant_params)
         
         # Extract loop parameters (consider making these instance vars of Alpaca if preferred)
         duration = assistant.duration_arg
         timeout = assistant.timeout_arg
         phrase_limit = assistant.phrase_limit_arg

         print(f"Starting main loop with timeout={timeout}, phrase_limit={phrase_limit}, duration={duration}")

         # --- Main Interaction Loop (Now using await) --- 
         while True:
              # Use await to call the async interaction handler
              user_input_status, assistant_output = await assistant.interaction_handler.run_single_interaction(
                   duration=duration,
                   timeout=timeout,
                   phrase_limit=phrase_limit
              )
              
              # Handle interaction status
              if user_input_status == "ERROR":
                   print(f"Recovering from interaction error: {assistant_output}")
                   # Use asyncio.sleep in async main
                   await asyncio.sleep(2) 
              elif user_input_status == "INTERRUPTED":
                   print("Interaction interrupted, starting new loop.")
              # COMPLETED/DISABLED/etc. statuses just continue the loop

    except KeyboardInterrupt:
         print("\nKeyboard interrupt detected. Attempting to summarize session before exiting...")
         # --- Execute Summarization Logic Here --- 
         if assistant: # Check if assistant was successfully initialized
             print("--- Running Session Summarization --- ")
             if not hasattr(assistant, 'conversation_manager') or \
                not hasattr(assistant, 'component_manager') or \
                not hasattr(assistant.component_manager, 'llm_handler'):
                 print("[Summarizer] Assistant or required managers/handlers not available. Cannot summarize.")
             else:
                 conv_manager = assistant.conversation_manager
                 llm_handler_inst = assistant.component_manager.llm_handler
                 history = conv_manager.get_history()
                 
                 if not history:
                     print("[Summarizer] No conversation history to summarize.")
                 else:
                     # Define the async operation we want to run
                     async def do_summarization():
                          return await summarize_conversation(history, llm_handler_inst)
                 
                     # Run the async summarization synchronously using the helper
                     summary = run_async_in_sync_context(do_summarization())
                 
                     if summary and summary not in ("[Error generating summary]", "[Error during async execution]"):
                         # --- Pass data_path_value to save_summary --- 
                         save_summary(summary, len(history), base_data_path=data_path_value) 
                         # -------------------------------------------
                     else:
                         print("[Summarizer] Failed to generate or execute summary.")
             print("--- Session Summarization Finished ---")
         else:
             print("[Summarizer] Assistant object not available, cannot summarize.")
         # --- End Summarization Logic --- 

    except Exception as e:
         print(f"\nAn unexpected fatal error occurred in the main execution: {e}")
         traceback.print_exc()
         # Consider if summarization should happen here too for unexpected errors
         # For now, only doing it on KeyboardInterrupt
    finally:
         # --- Cleanup --- 
         print("Performing final cleanup...") 
         if assistant and hasattr(assistant, 'component_manager'):
              try:
                  assistant.component_manager.cleanup()
              except Exception as cleanup_e:
                   # Catch potential errors during cleanup (like the double free)
                   print(f"Error during component cleanup: {cleanup_e}")
                   traceback.print_exc()
         print("AI Voice assistant shut down process complete.")

if __name__ == "__main__":
    # Run the async main function using asyncio.run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handling potential KeyboardInterrupt during asyncio.run itself
        print("\nAsyncio run interrupted. Exiting.")
    except Exception as e:
        print(f"\nUnhandled exception during asyncio execution: {e}")
        traceback.print_exc()