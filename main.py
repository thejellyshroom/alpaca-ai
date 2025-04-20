import sys
import os
import asyncio
import signal # Import signal module
import traceback
from dotenv import load_dotenv

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
from src.utils.session_utils import summarize_conversation, save_summary


# Change main to async def
async def main():
    # Load environment variables from .env file first
    load_dotenv()
    print("Initializing AI Voice assistant...")
    assistant = None # Initialize assistant to None for finally block
    current_task = None # Variable to hold the current interaction task
    shutdown_requested = False # Flag for graceful shutdown
    
    # --- Get DATA_PATH early --- 
    data_path_value = os.getenv("DATA_PATH", "./data/dataset") # Get path, provide default
    print(f"Using DATA_PATH: {data_path_value}")
    # ---------------------------
    
    # --- Define Signal Handler --- 
    def handle_shutdown_signal(*args):
        nonlocal shutdown_requested, current_task
        if not shutdown_requested:
             print("\nShutdown signal received. Initiating graceful shutdown...")
             shutdown_requested = True
             # Cancel the currently running interaction task, if any
             if current_task and not current_task.done():
                 current_task.cancel()
        else:
            print("Shutdown already in progress.")

    # --- RAG Indexing (Now awaited) --- 
    try:
        print("--- Running RAG Indexing --- ")
        await run_indexing() # Use await
        print("--- RAG Indexing Complete --- \n")
    except Exception as e:
        print(f"***** CRITICAL ERROR DURING RAG INDEXING *****: {e}")
        print("***** RAG features may be unavailable or outdated. *****")
        traceback.print_exc()
    # -----------------------------------
    config_loader = ConfigLoader()
    assistant_params = config_loader.load_all()

    if assistant_params is None:
         print("Failed to load configurations. Exiting.")
         sys.exit(1)

    try:
         assistant = Alpaca(**assistant_params)
         
         duration = assistant.duration_arg
         timeout = assistant.timeout_arg
         phrase_limit = assistant.phrase_limit_arg

         # --- Register Signal Handler --- 
         loop = asyncio.get_running_loop()
         loop.add_signal_handler(signal.SIGINT, handle_shutdown_signal)

         # --- Main Interaction Loop --- 
         while not shutdown_requested:
              if current_task and not current_task.done():
                   await asyncio.sleep(0.5) # Wait briefly before starting new task
                   continue
                   
              try:
                  current_task = asyncio.create_task(
                       assistant.interaction_handler.run_single_interaction(
                           duration=duration,
                           timeout=timeout,
                           phrase_limit=phrase_limit
                       )
                  )
                  user_input_status, assistant_output = await current_task
                  
                  if user_input_status == "ERROR":
                       print(f"Recovering from interaction error: {assistant_output}")
                       await asyncio.sleep(2) 
                  current_task = None 

              except asyncio.CancelledError:
                   print("Interaction task cancelled due to shutdown request.")
                   shutdown_requested = True
                   break 
              
              except Exception as loop_e:
                   print(f"\nError during interaction loop iteration: {loop_e}")
                   traceback.print_exc()
                   current_task = None # Clear task on other exceptions too
                   await asyncio.sleep(2) 
         # --- End Main Interaction Loop --- 

         print("Main loop exited gracefully.") # Add confirmation

    except Exception as e:
         # Catch errors during setup
         print(f"\nAn unexpected fatal error occurred during setup or loop: {e}")
         traceback.print_exc()
    
    finally:
        try:
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                loop.remove_signal_handler(signal.SIGINT)
                print("Signal handler removed.")
        except RuntimeError:
             print("No running event loop to remove signal handler from.")
        except Exception as e_remove:
            print(f"Error removing signal handler: {e_remove}")
        
        # --- Summarization on Graceful Shutdown --- 
        if shutdown_requested and assistant: 
            print("\n--- Running Session Summarization (Graceful Shutdown) --- ")
            if hasattr(assistant, 'conversation_manager') and \
               hasattr(assistant, 'component_manager') and \
               hasattr(assistant.component_manager, 'llm_handler'):
                
                conv_manager = assistant.conversation_manager
                llm_handler_inst = assistant.component_manager.llm_handler
                history = conv_manager.get_history()
                
                if not history:
                    print("[Summarizer] No conversation history to summarize.")
                else:
                    try:
                        # Await the async summarization directly
                        print("[Summarizer] Generating summary...")
                        summary = await summarize_conversation(history, llm_handler_inst)
                        if summary and summary != "[Error generating summary]":
                            save_summary(summary, len(history), base_data_path=data_path_value) 
                        else:
                            print("[Summarizer] Failed to generate summary.")
                    except Exception as summary_e:
                        print(f"[Summarizer] Error during summarization/saving: {summary_e}")
                        traceback.print_exc()
            else:
                print("[Summarizer] Assistant components missing, cannot summarize.")
            print("--- Session Summarization Finished --- ")
        # --------------------------------------------

        # --- Component Cleanup --- 
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
    # Keep the asyncio.run call
    try:
        asyncio.run(main())
    except Exception as e:
        # Catch errors that might prevent asyncio.run from completing
        print(f"\nUnhandled exception during asyncio execution: {e}")
        traceback.print_exc()
    finally:
        print("Application exiting.")