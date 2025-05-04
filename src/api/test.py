import asyncio
import websockets
import json
import base64
import numpy as np
import sounddevice as sd
import traceback

async def test_interaction(mode="text", text_to_send=None):
    uri = "ws://localhost:8000/ws" # Make sure the port matches your uvicorn command
    audio_buffer = [] # Buffer to store incoming audio chunks
    sample_rate = 16000 # Default sample rate, update if received
    audio_format = 'int16' # Default format based on pcm_s16le

    print(f"--- Testing {mode.upper()} mode ---")
    if mode == "text" and not text_to_send:
        text_to_send = "Hello from the test script! Tell me about Alpacas."

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")

            # --- Prepare Message --- 
            if mode == "voice":
                message_to_send = {
                    "action": "start",
                    "mode": "voice"
                }
            elif mode == "text":
                message_to_send = {
                    "action": "send_text",
                    "text": text_to_send
                }
            else:
                print(f"Error: Unsupported test mode '{mode}'")
                return
            # ---------------------

            print(f"> Sending: {json.dumps(message_to_send)}")
            await websocket.send(json.dumps(message_to_send))

            print("< Receiving messages...")
            try:
                while True:
                    response = await websocket.recv()
                    print(f"< Received raw: {response[:100]}...") # Print truncated raw response

                    try:
                        data = json.loads(response)
                        msg_type = data.get("type")
                        state = data.get("state")
                        
                        print(f"< Parsed Type: {msg_type}, State: {state}") # Log parsed type/state

                        if msg_type == "audio_chunk":
                            base64_audio = data.get("data")
                            received_rate = data.get("sample_rate")
                            received_format = data.get("format", "pcm_s16le").lower()

                            if received_rate:
                                sample_rate = int(received_rate)
                            
                            # Determine numpy dtype based on format
                            dtype = np.int16 # Default for pcm_s16le
                            if "f32" in received_format: # Example check for float32
                                dtype = np.float32
                                print(f"    (Audio format: {dtype})")
                            # Add more checks if other formats are possible

                            if base64_audio:
                                try:
                                    audio_bytes = base64.b64decode(base64_audio)
                                    # Convert bytes to numpy array
                                    audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                                    print(f"    Decoded {len(audio_bytes)} audio bytes, playing {len(audio_array)} samples at {sample_rate} Hz...")
                                    sd.play(audio_array, samplerate=sample_rate)
                                    # --- Wait for playback to finish before processing next message --- 
                                    print("    Waiting for chunk playback to complete...")
                                    sd.wait() 
                                    print("    Playback complete.")
                                    # --- End wait ---

                                except base64.binascii.Error as b64e:
                                     print(f"    Error decoding base64: {b64e}")
                                except Exception as play_e:
                                     print(f"    Error playing audio chunk: {play_e}")
                                     traceback.print_exc()
                        
                        elif msg_type == "status" and state in ["Idle", "Error", "Interrupted", "Cancelled", "Disabled"]:
                            print(f"<- Received final state '{state}'. Waiting for audio playback...")
                            sd.wait() # Wait for any remaining audio to finish playing
                            print("<- Playback finished. Closing connection.")
                            break # Exit loop
                        
                        # Print other message types like status changes, transcripts etc.
                        elif msg_type != "audio_chunk": 
                             print(f"< Received JSON: {data}") 

                    except json.JSONDecodeError:
                        print(f"< Received non-JSON message: {response[:100]}...")
                    except Exception as e:
                        print(f"< Error processing received message: {e}")
                        traceback.print_exc()

            except websockets.exceptions.ConnectionClosedOK:
                print("< Connection closed normally.")
                sd.wait() # Ensure audio finishes if connection closed mid-stream
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"< Connection closed with error: {e}")
                sd.wait()

    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is the server running at {uri}?")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        print("Stopping any lingering audio playback...")
        sd.stop()

if __name__ == "__main__":
    # Ensure the server is running before executing this script
    # Run the FastAPI server: uvicorn src.api.server:app --reload --port 8000
    print("--- Make sure the FastAPI server is running! ---")
    # asyncio.run(test_interaction(mode="text"))
    asyncio.run(test_interaction(mode="voice"))
