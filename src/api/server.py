import asyncio
import os
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Union

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
# Keep track of active connections, VAD settings, etc. for the WS endpoint
# E.g., active_connection: WebSocket | None = None
# E.g., is_vad_interrupt_enabled: bool = True # Default VAD setting per connection might be better
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
    #     print("--- RAG Indexing Complete --- \\n")
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
        print("Initializing Alpaca instance...")
        alpaca_instance = Alpaca(**loaded_config_data, mode='api') # Using 'api' mode
        print("Alpaca instance initialized successfully for API.")
    except Exception as e:
        print(f"FATAL: Error initializing Alpaca instance: {e}")
        traceback.print_exc()
        alpaca_instance = None # Ensure instance is None if init fails
        # Prevent startup or run in a degraded state? For now, allow startup but endpoints will fail.
    # ------------------------
    print("API Server startup complete.")
    # pass # Remove pass

@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources on server shutdown."""
    global alpaca_instance, loaded_config_data
    print("API Server shutting down...")
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the main WebSocket connection for real-time interaction."""
    global alpaca_instance # Need access to the initialized instance
    # Per-connection state variables could go here if needed
    # E.g., is_vad_interrupt_enabled = True
    # E.g., current_interaction_task: asyncio.Task | None = None

    await websocket.accept()
    print("WebSocket connection established.")

    # --- Simplification: Allow only one connection at a time globally ---
    # In a real app, you'd manage connections and state per client.
    # For now, we assume one user.
    # ------------------------------------------------------------------

    # --- Send Initial State (Optional but good practice) ---
    # await websocket.send_json({
    #     "type": "status",
    #     "state": "Idle",
    #     "vad_interrupt_enabled": True # Default or fetch from config
    # })
    # --------------------------------------------------------

    # Task tracking for cancellation on disconnect or stop
    current_interaction_task: Union[asyncio.Task, None] = None

    try:
        while True:
            data = await websocket.receive_json()
            print(f"Received WS message: {data}")

            action = data.get("action")

            # Check if Alpaca is initialized
            if not alpaca_instance:
                await websocket.send_json({"type": "error", "message": "Alpaca assistant not initialized.", "state": "Error"})
                continue # Wait for next message

            # --- Action Handling ---
            if action == "start":
                mode = data.get("mode", "voice")
                print(f"Received 'start' action, mode: {mode}")
                # TODO: Implement voice start
                await websocket.send_json({"type": "status", "state": "Starting", "message": f"Starting {mode} mode... (Voice not implemented yet)"})

            elif action == "stop":
                print("Received 'stop' action")
                # TODO: Cancel the current_interaction_task if it's running
                # TODO: Send 'Idle' status update
                await websocket.send_json({"type": "status", "state": "Stopping"})

            elif action == "send_text":
                text = data.get("text")
                if not text:
                    await websocket.send_json({"type": "error", "message": "Received empty text for 'send_text' action.", "state": "Idle"})
                    continue

                print(f"Received 'send_text': '{text[:50]}...'")
                await websocket.send_json({"type": "status", "state": "Processing"})

                try:
                    # Ensure interaction_handler exists
                    if not hasattr(alpaca_instance, 'interaction_handler'):
                        raise AttributeError("Alpaca instance lacks an 'interaction_handler'")

                    # Call the async method which returns an async generator
                    response_generator = await alpaca_instance.interaction_handler.run_single_text_interaction(text)

                    full_response = ""
                    # Stream the response back chunk by chunk
                    # --- Use standard 'for' loop for sync generator, but yield control --- 
                    for chunk in response_generator: # Use standard 'for'
                        if chunk: # Ensure chunk is not empty
                            full_response += chunk
                            await websocket.send_json({ # Still await WS send
                                "type": "llm_chunk",
                                "text": chunk
                            })
                        await asyncio.sleep(0) # Yield control to event loop
                    # --- End iteration ---
                    
                    # Send a final status update once streaming is complete
                    await websocket.send_json({
                        "type": "status", 
                        "state": "Idle", 
                        "final_response": full_response # Optional: Send full response at the end
                    }) 
                    print("Text interaction streaming complete.")
                
                except AttributeError as ae:
                     print(f"Error accessing interaction handler: {ae}")
                     traceback.print_exc()
                     await websocket.send_json({"type": "error", "message": f"Server configuration error: {ae}", "state": "Error"})
                except Exception as e:
                    print(f"Error during text interaction: {e}")
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": f"Error processing text: {e}", "state": "Error"})
                    # Optionally revert to Idle state after error
                    await websocket.send_json({"type": "status", "state": "Idle"})


            elif action == "interrupt":
                print("Received 'interrupt' action")
                # TODO: Implement interrupt (mainly for voice)
                await websocket.send_json({"type": "status", "state": "Interrupted"})

            elif action == "toggle_vad_interrupt":
                enabled = data.get("enabled", False)
                print(f"Received 'toggle_vad_interrupt', enabled: {enabled}")
                # TODO: Implement VAD toggle state management
                await websocket.send_json({"type": "info", "message": f"VAD Interrupt Toggled: {enabled}"})

            else:
                print(f"Unknown action received: {action}")
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})
            # --- End Action Handling ---

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
        # TODO: Handle disconnection - cancel any running tasks for this client
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": f"Server error: {e}", "state": "Error"})
            await websocket.close(code=1011)
        except Exception:
            pass # Ignore if sending/closing fails
    finally:
        # --- Cleanup for this connection ---
        print("WebSocket cleanup complete.")

# --- Optional: Add entry point for running with uvicorn ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting server with uvicorn...")
#     # Remember to set PYTHONPATH=. or similar if running directly
#     # Or run using: uvicorn src.api.server:app --reload --port 8000
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# --------------------------------------------------------- 