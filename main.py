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

import asyncio
import signal
import traceback
import argparse
import sys
from dotenv import load_dotenv

from core.alpaca import Alpaca
from utils.config_loader import ConfigLoader
from core.voice_loop import run_voice_interaction_loop
from core.text_loop import run_text_interaction_loop
from rag.indexer import run_indexing
from utils.summarizer import summarize_conversation, save_summary
    
async def main():
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
    main_task = None
    shutdown_requested = False
    
    data_path_value = os.getenv("DATA_PATH", "./data/dataset")
    
    def handle_shutdown_signal(task_to_cancel: asyncio.Task):
        nonlocal shutdown_requested
        if not shutdown_requested:
             print("\nShutdown signal received...")
             shutdown_requested = True
             if task_to_cancel and not task_to_cancel.done():
                 task_to_cancel.cancel()
        else:
            print("Shutdown already in progress.")

    # --- RAG Indexing --- 
    try:
        print("--- Running RAG Indexing --- ")
        await run_indexing()
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

        # --- Register Signal Handler --- 
        loop = asyncio.get_running_loop()

        # --- Create and Run Main Loop Task --- 
        if run_mode == 'voice':
            main_task = asyncio.create_task(
                run_voice_interaction_loop(assistant, duration, timeout, phrase_limit),
                name="VoiceInteractionLoop"
            )
        elif run_mode == 'text':
            main_task = asyncio.create_task(
                run_text_interaction_loop(assistant),
                name="TextInteractionLoop"
            )
        else:
            print(f"Error: Invalid run mode '{run_mode}'")
            sys.exit(1)

        # Now that main_task exists, add the signal handler
        loop.add_signal_handler(signal.SIGINT, handle_shutdown_signal, main_task)

        await main_task
         # --- End Main Loop Task --- 
        print("Main loop exited.")

    except Exception as e:
         print(f"\nAn unexpected fatal error occurred during setup or loop: {e}")
         traceback.print_exc()
    
    finally:
        try:
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except RuntimeError:
                     print("No running event loop to remove signal handler from.")
                except Exception as e_remove:
                    print(f"Error removing signal handler: {e_remove}")
        except RuntimeError:
             print("No running event loop to remove signal handler from.")
        except Exception as e_remove:
            print(f"Error removing signal handler: {e_remove}")
        
        # --- Summarization on Graceful Shutdown --- 
        if shutdown_requested and assistant: 
            print("\n--- Running Session Summarization --- ")
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
                        summary = await summarize_conversation(history, llm_handler_inst)
                        if summary:
                            save_summary(summary, len(history), base_data_path=data_path_value) 
                    except Exception as summary_e:
                        print(f"[Summarizer] Error during summarization/saving: {summary_e}")
                        traceback.print_exc()
            else:
                print("[Summarizer] Assistant components missing, cannot summarize.")
            print("--- Session Summarization Finished --- ")
        # --------------------------------------------

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
        asyncio.run(main())
    except Exception as e:
        print(f"\nUnhandled exception during asyncio execution: {e}")
        traceback.print_exc()
    finally:
        print("Application exiting.")