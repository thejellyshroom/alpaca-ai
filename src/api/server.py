import asyncio
import os
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Union, Optional
from asyncio import Queue

# --- Add project root to sys.path ---
# This allows importing modules from src, utils, etc.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# add rag to sys.path
rag_path = os.path.join(project_root, 'src/rag')
if rag_path not in sys.path:
    sys.path.insert(0, rag_path)
# --- Imports from your project ---
from core.alpaca import Alpaca
from utils.config_loader import ConfigLoader
# from core.alpaca_interaction import AlpacaInteraction # Might be needed later
import traceback # For error logging
from dotenv import load_dotenv
# ---------------------------------

# --- Globals ---
# This will hold the initialized Alpaca instance
alpaca_instance: Union[Alpaca, None] = None
# This will hold the configuration loaded at startup
loaded_config_data: Union[Dict[str, Any], None] = None
# ----------------

app = FastAPI(
    title="Alpaca Voice Assistant API",
    description="API endpoints for controlling and interacting with the Alpaca voice assistant.",
    version="0.1.0"
)

# --- State Variables ---
# For simplicity, we manage task state globally for a single connection.
# In a multi-user scenario, this state would need to be managed per WebSocket connection.
current_interaction_task: Optional[asyncio.Task] = None
queue_reader_task: Optional[asyncio.Task] = None
interaction_queue: Optional[Queue] = None
# ---------------------

@app.on_event("startup")
async def startup_event():
    """Loads configuration and initializes the Alpaca instance on server start."""
    global alpaca_instance, loaded_config_data
    print("API Server starting up...")
    load_dotenv() # Load .env file for configurations

    # --- RAG Indexing (Optional but recommended, similar to main.py) ---
    # You might want to run indexing here if the API needs up-to-date RAG data at startup
    # try:
    #     print("--- Running RAG Indexing --- ")
    #     from rag.indexer import run_indexing # Import locally if run here
    #     await run_indexing()
    #     print("--- RAG Indexing Complete --- \n")
    # except Exception as e:
    #     print(f"***** WARNING: ERROR DURING RAG INDEXING *****: {e}")
    #     print("***** RAG features may be unavailable or outdated. *****")
    #     traceback.print_exc()
    # -------------------------------------------------------------------

    # --- Load Configuration ---
    try:
        config_loader = ConfigLoader()
        # Pass specific paths if necessary, otherwise uses defaults / env vars
        assistant_params = config_loader.load_all()
        if not assistant_params:
            print("FATAL: Failed to load configurations for API server. Check config files and .env")
            return
        loaded_config_data = assistant_params # Store loaded config globally
        print("Configurations loaded.")
    except Exception as e:
        print(f"FATAL: Error loading configurations: {e}")
        traceback.print_exc()
        return # Prevent startup if config fails
    # -------------------------

    # --- Initialize Alpaca ---
    try:
        # Initialize Alpaca. Using 'api' mode conceptually.
        # The mode might influence which components are strictly required
        # or how certain loops behave, though interaction is driven by WS.
        # We might need to adjust Alpaca.__init__ if 'api' mode needs specific handling.
        # For now, assume it loads necessary components based on config.
        # --- CORRECTION: Need 'voice' mode to load audio components for voice interactions --- 
        print("Initializing Alpaca instance (mode='voice')...")
        alpaca_instance = Alpaca(**loaded_config_data, mode='voice') # Use 'voice' mode
        print("Alpaca instance initialized successfully for API.")
    except Exception as e:
        print(f"FATAL: Error initializing Alpaca instance: {e}")
        traceback.print_exc()
        alpaca_instance = None # Ensure instance is None if init fails
        # Prevent startup or run in a degraded state? For now, allow startup but endpoints will fail.
    # ------------------------
    print("API Server startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources on server shutdown."""
    global alpaca_instance, loaded_config_data, current_interaction_task, queue_reader_task
    print("API Server shutting down...")
    # --- Cancel any running tasks ---
    if current_interaction_task and not current_interaction_task.done():
        print("Cancelling active interaction task...")
        current_interaction_task.cancel()
    if queue_reader_task and not queue_reader_task.done():
        print("Cancelling queue reader task...")
        queue_reader_task.cancel()
    # -----------------------------
    # --- Cleanup Alpaca Components ---
    if alpaca_instance and hasattr(alpaca_instance, 'component_manager'):
        print("Cleaning up Alpaca components...")
        try:
            # Assuming component_manager.cleanup() handles stopping threads/processes etc.
            # If cleanup needs to be async, adjust Alpaca/ComponentManager accordingly.
            # For now, assuming it's synchronous or handles async internally.
            alpaca_instance.component_manager.cleanup()
            print("Alpaca components cleaned up.")
        except Exception as e:
            print(f"Error during Alpaca component cleanup: {e}")
            traceback.print_exc()
    elif alpaca_instance:
        print("Alpaca instance exists but has no component_manager attribute for cleanup.")
    else:
        print("No Alpaca instance to clean up.")
    # --------------------------------
    # Clear global state
    alpaca_instance = None
    loaded_config_data = None
    current_interaction_task = None
    queue_reader_task = None
    print("API Server shutdown complete.")


@app.get("/config", response_model=Dict[str, Any])
async def get_config():
    """Returns the current configuration loaded by the Alpaca assistant at startup."""
    global loaded_config_data
    if loaded_config_data:
        # Return a copy to prevent accidental modification if needed, though FastAPI handles serialization
        return JSONResponse(content=loaded_config_data.copy())
    else:
        # Return 503 Service Unavailable if config wasn't loaded during startup
        return JSONResponse(
            content={"error": "Configuration not available. Server may not have started correctly."},
            status_code=503
        )

# --- WebSocket Helper ---
async def handle_interaction_queue(websocket: WebSocket, queue: Queue):
    """Reads messages from the interaction queue and sends them to the client."""
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
            queue.task_done()
            # If the message indicates the end of interaction (e.g., Idle, Error, Interrupted, Cancelled, Disabled), stop reading
            if message.get("type") == "status" and message.get("state") in ["Error", "Cancelled", "Disabled"]:
                 print(f"[QueueReader] Received final state '{message.get('state')}'. Exiting.")
                 break
    except asyncio.CancelledError:
        print("[QueueReader] Task cancelled.")
    except WebSocketDisconnect:
        print("[QueueReader] WebSocket disconnected.")
    except Exception as e:
        print(f"[QueueReader] Error: {e}")
        traceback.print_exc()
        # Try to send error to client if possible
        try:
            await websocket.send_json({"type": "error", "message": f"Queue reader error: {e}", "state": "Error"})
        except:
            pass # Ignore if sending fails
    finally:
        print("[QueueReader] Exiting task.")

# --- End WebSocket Helper ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the main WebSocket connection for real-time interaction."""
    global alpaca_instance, current_interaction_task, queue_reader_task, interaction_queue

    client_address = f"{websocket.client.host}:{websocket.client.port}"
    print(f"WebSocket connection established from {client_address}")

    # --- Simple single-client handling ---
    # If an interaction is somehow active from a previous connection, try to cancel it.
    if current_interaction_task and not current_interaction_task.done():
        print(f"Warning: Cancelling remnant interaction task from previous connection.")
        current_interaction_task.cancel()
    if queue_reader_task and not queue_reader_task.done():
        print(f"Warning: Cancelling remnant queue reader task from previous connection.")
        queue_reader_task.cancel()
    current_interaction_task = None
    queue_reader_task = None
    interaction_queue = None # Reset queue on new connection
    # -------------------------------------

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            print(f"Received WS message: {data}")

            action = data.get("action")

            if not alpaca_instance:
                await websocket.send_json({"type": "error", "message": "Alpaca assistant not initialized.", "state": "Error"})
                continue

            # --- Action Handling ---
            if action == "start":
                mode = data.get("mode", "voice")
                print(f"Received 'start' action, mode: {mode}")

                if current_interaction_task and not current_interaction_task.done():
                     await websocket.send_json({"type": "error", "message": "An interaction is already in progress.", "state": "Busy"})
                     continue

                if mode == "voice":
                    # --- Start Voice Interaction ---
                    try:
                        interaction_queue = Queue()

                        # Start the task to read from the queue and send to websocket
                        queue_reader_task = asyncio.create_task(
                            handle_interaction_queue(websocket, interaction_queue),
                            name=f"QueueReader_{client_address}"
                        )

                        # Fetch interaction parameters (optional, use defaults or pass via WS)
                        timeout = alpaca_instance.timeout_arg if hasattr(alpaca_instance, 'timeout_arg') else 10
                        phrase_limit = alpaca_instance.phrase_limit_arg if hasattr(alpaca_instance, 'phrase_limit_arg') else 10
                        duration = alpaca_instance.duration_arg if hasattr(alpaca_instance, 'duration_arg') else None
                        
                        print(f"Starting voice interaction task (timeout={timeout}, phrase_limit={phrase_limit}, duration={duration})...")
                        # Start the actual interaction task, passing the queue
                        current_interaction_task = asyncio.create_task(
                            alpaca_instance.interaction_handler.run_voice_interaction_loop(
                                status_queue=interaction_queue,
                                duration=duration,
                                timeout=timeout,
                                phrase_limit=phrase_limit
                            ),
                            name=f"VoiceInteractionLoop_{client_address}"
                        )

                        # Optional: Monitor the interaction task completion/failure
                        # You could add a callback or await it here, but that blocks receive loop.
                        # The handle_interaction_queue task will exit based on final status messages.
                        # Consider adding logic to handle task exceptions if needed.

                    except AttributeError as ae:
                         print(f"Error accessing alpaca instance attributes for voice start: {ae}")
                         traceback.print_exc()
                         await websocket.send_json({"type": "error", "message": f"Server configuration error: {ae}", "state": "Error"})
                         # Clean up queue/tasks if partially created
                         if queue_reader_task and not queue_reader_task.done(): queue_reader_task.cancel()
                         if interaction_queue: interaction_queue = None
                         queue_reader_task = None
                         current_interaction_task = None
                    except Exception as e:
                         print(f"Error starting voice interaction: {e}")
                         traceback.print_exc()
                         await websocket.send_json({"type": "error", "message": f"Failed to start interaction: {e}", "state": "Error"})
                         if queue_reader_task and not queue_reader_task.done(): queue_reader_task.cancel()
                         if interaction_queue: interaction_queue = None
                         queue_reader_task = None
                         current_interaction_task = None
                    # --- End Start Voice Interaction ---

                elif mode == "text":
                     # This assumes text interaction is short and doesn't need complex task management
                     # Re-using existing text logic here
                     text = data.get("text", "") # Allow text with start action? Or require send_text? Let's assume require send_text.
                     await websocket.send_json({"type": "info", "message": "Use 'send_text' action for text interactions."})
                     # If you want start to trigger a text loop, implement similar task logic as voice
                else:
                     await websocket.send_json({"type": "error", "message": f"Unsupported start mode: {mode}", "state": "Error"})


            elif action == "stop":
                print("Received 'stop' action")
                interrupted_by_client = False
                if current_interaction_task and not current_interaction_task.done():
                    print("Cancelling interaction task due to 'stop' command.")
                    current_interaction_task.cancel()
                    # Wait briefly for cancellation to propagate if needed, though not strictly necessary
                    # await asyncio.sleep(0.01)
                    interrupted_by_client = True
                else:
                     print("No active interaction task to stop.")
                
                # Queue reader task should exit automatically when it receives the final
                # status ('Cancelled' or 'Interrupted') from the queue after the main task cancels,
                # or on WebSocketDisconnect.
                # We don't need to explicitly cancel queue_reader_task here unless it gets stuck.

                # Reset global task/queue variables after cancellation attempt
                current_interaction_task = None
                queue_reader_task = None # Let it finish naturally
                interaction_queue = None # Clear queue reference

                # Send status based on whether we cancelled something
                if interrupted_by_client:
                    # The interaction task will put 'Cancelled' or 'Interrupted' on the queue,
                    # which handle_interaction_queue will send.
                    # Sending an immediate status might be redundant or confusing.
                    # Let's just confirm the stop was processed.
                    await websocket.send_json({"type": "info", "message": "Stop command processed. Interaction cancelled."})
                    # Optional: Send Idle state if confident cancellation worked immediately
                    # await websocket.send_json({"type": "status", "state": "Idle", "message": "Stopped by client."}) 
                else:
                     await websocket.send_json({"type": "status", "state": "Idle", "message": "Stop command received, nothing active to stop."})


            elif action == "send_text":
                text = data.get("text")
                if not text:
                    await websocket.send_json({"type": "error", "message": "Received empty text for 'send_text' action.", "state": "Idle"})
                    continue

                print(f"Received 'send_text': '{text[:50]}...'")
                
                if current_interaction_task and not current_interaction_task.done():
                    await websocket.send_json({"type": "error", "message": "Cannot send text while voice interaction is active.", "state": "Busy"})
                    continue

                await websocket.send_json({"type": "status", "state": "Processing"})
                try:
                    if not hasattr(alpaca_instance, 'interaction_handler'):
                        raise AttributeError("Alpaca instance lacks an 'interaction_handler'")
                    response_generator = await alpaca_instance.interaction_handler.run_single_text_interaction(text)
                    full_response = ""
                    for chunk in response_generator:
                        if chunk:
                            full_response += chunk
                            await websocket.send_json({"type": "llm_chunk", "text": chunk})
                        await asyncio.sleep(0)
                    await websocket.send_json({"type": "status", "state": "Idle", "final_response": full_response})
                    print("Text interaction streaming complete.")
                except AttributeError as ae:
                     print(f"Error accessing interaction handler: {ae}")
                     traceback.print_exc()
                     await websocket.send_json({"type": "error", "message": f"Server configuration error: {ae}", "state": "Error"})
                except Exception as e:
                    print(f"Error during text interaction: {e}")
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": f"Error processing text: {e}", "state": "Error"})
                    await websocket.send_json({"type": "status", "state": "Idle"})


            elif action == "interrupt":
                print("Received 'interrupt' action")
                interrupted_tts = False
                # Check if alpaca and handlers exist
                if alpaca_instance and hasattr(alpaca_instance, 'output_handler'):
                    output_handler = alpaca_instance.output_handler
                    if hasattr(output_handler, 'interrupt') and callable(output_handler.interrupt):
                        try:
                            print("Calling output_handler.interrupt()...")
                            output_handler.interrupt() # Call the new method
                            interrupted_tts = True
                        except Exception as e:
                            print(f"Error calling output_handler.interrupt(): {e}")
                    else:
                        print("Warning: Output handler does not have an interrupt() method.")
                else:
                    print("Warning: Cannot interrupt TTS, Alpaca instance or Output handler missing.")

                # Send confirmation back to client
                if interrupted_tts:
                    # The actual status change (Interrupted, Idle) will come via the queue later
                    await websocket.send_json({"type": "info", "message": "Interrupt signal sent to TTS handler."})
                else:
                    await websocket.send_json({"type": "info", "message": "Interrupt received, but TTS handler could not be signalled."})


            elif action == "toggle_vad_interrupt":
                # TODO: Implement actual VAD toggle logic
                # This requires state management and potentially modifying OutputHandler/AudioHandler
                enabled = data.get("enabled", False)
                print(f"Received 'toggle_vad_interrupt', enabled: {enabled} (Logic not fully implemented)")
                # Example: Store state per connection if managing multiple clients
                # connection_state['vad_enabled'] = enabled
                await websocket.send_json({"type": "info", "message": f"VAD Interrupt Toggled: {enabled} (Server logic TBD)"})

            else:
                print(f"Unknown action received: {action}")
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})
            # --- End Action Handling ---

    except WebSocketDisconnect:
        print(f"WebSocket disconnected from {client_address}.")
        # Clean up tasks associated with this connection
        if current_interaction_task and not current_interaction_task.done():
            print("Cancelling interaction task due to disconnect.")
            current_interaction_task.cancel()
        if queue_reader_task and not queue_reader_task.done():
            print("Cancelling queue reader task due to disconnect.")
            queue_reader_task.cancel()
        current_interaction_task = None
        queue_reader_task = None
        interaction_queue = None

    except Exception as e:
        print(f"Error in WebSocket handler for {client_address}: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": f"Server error: {e}", "state": "Error"})
            await websocket.close(code=1011)
        except Exception:
            pass # Ignore if sending/closing fails
    finally:
        # Ensure cleanup if connection closes unexpectedly
        if current_interaction_task and not current_interaction_task.done(): current_interaction_task.cancel()
        if queue_reader_task and not queue_reader_task.done(): queue_reader_task.cancel()
        current_interaction_task = None
        queue_reader_task = None
        interaction_queue = None
        print(f"WebSocket cleanup complete for {client_address}.")

# --- Optional: Add entry point for running with uvicorn ---
if __name__ == "__main__":
    import uvicorn
    print("Starting server with uvicorn...")
    # Remember to set PYTHONPATH=. or similar if running directly
    # Or run using: uvicorn src.api.server:app --reload --port 8000 --log-level debug
    # Note: Uvicorn might need host='127.0.0.1' instead of '0.0.0.0' on some systems for localhost tests
    # Ensure the app object is referenced correctly if running this file directly
    # uvicorn.run("server:app", ...) should be uvicorn.run(__name__ + ":app", ...) or adjust depending on execution context
    # Let's make it runnable directly assuming file is run from project root with PYTHONPATH set
    # Or more robustly: uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload --log-level info
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info") # Removed reload for direct run 