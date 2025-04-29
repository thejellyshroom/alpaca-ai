import sys
import os

# --- Add src to sys.path --- Must be before src imports!
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# add rag to sys.path
rag_path = os.path.join(project_root, 'src/rag')
if rag_path not in sys.path:
    sys.path.insert(0, rag_path)
# ---------------------------

import signal
import traceback
import argparse
import sys
from dotenv import load_dotenv
import time

from core.alpaca import Alpaca
from utils.config_loader import ConfigLoader
from core.voice_loop import run_voice_interaction_loop
from core.text_loop import run_text_interaction_loop
from rag.indexer import run_indexing
from utils.summarizer import summarize_conversation, save_summary
    
def main():
    load_dotenv()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Alpaca AI Voice/Text Assistant")
    default_mode = os.getenv('DEFAULT_MODE', 'voice').lower()
    parser.add_argument(
        '--mode',
        type=str,
        choices=['voice', 'text'],
        default=default_mode,
        help="Run in 'voice' mode (voice input/output) or 'text' mode (text input/output). Default from DEFAULT_MODE env var."
    )
    args = parser.parse_args()
    run_mode = args.mode
    print(f"Running in {run_mode.upper()} mode.")
    # --- End Argument Parsing ---

    print("Initializing AI Voice assistant...")
    assistant = None
    
    data_path_value = os.getenv("DATA_PATH", "./data/dataset")
    
    # --- RAG Indexing --- 
    try:
        print("--- Running RAG Indexing --- ")
        run_indexing()
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
        assistant = Alpaca(**assistant_params, mode=run_mode)
        
        duration = assistant.duration_arg
        timeout = assistant.timeout_arg
        phrase_limit = assistant.phrase_limit_arg

        # --- Run Main Loop Synchronously --- 
        print("Starting main interaction loop...")
        if run_mode == 'voice':
            run_voice_interaction_loop(assistant, duration, timeout, phrase_limit)
        elif run_mode == 'text':
            run_text_interaction_loop(assistant)
        else:
            print(f"Error: Invalid run mode '{run_mode}'")
            sys.exit(1)

        print("Main loop finished (likely due to user exit request).")

    except KeyboardInterrupt:
         print("\nKeyboard interrupt detected by main. Initiating shutdown...")
         
    except Exception as e:
         print(f"\nAn unexpected fatal error occurred during setup or loop: {e}")
         traceback.print_exc()
    
    finally:
        print("\n--- Attempting Session Summarization --- ")
        if assistant: 
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
                        summary = summarize_conversation(history, llm_handler_inst)
                        if summary:
                            save_summary(summary, len(history), base_data_path=data_path_value) 
                    except Exception as summary_e:
                        print(f"[Summarizer] Error during summarization/saving: {summary_e}")
                        traceback.print_exc()
            else:
                print("[Summarizer] Assistant components missing, cannot summarize.")
        else:
            print("[Summarizer] Assistant object not created, cannot summarize.")
        print("--- Session Summarization Finished --- ")
        
        print("Performing final cleanup...") 
        if assistant and hasattr(assistant, 'component_manager'):
            try:
                assistant.component_manager.cleanup()
            except Exception as cleanup_e:
                print(f"Error during component cleanup: {cleanup_e}")
                traceback.print_exc()
        print("AI Voice assistant shut down process complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nUnhandled top-level exception during execution: {e}")
        traceback.print_exc()
    finally:
        print("Application exiting.")