import asyncio
import traceback

async def run_voice_interaction_loop(assistant, duration, timeout, phrase_limit):
    """Runs the main interaction loop for voice mode."""
    print("Starting VOICE mode interaction loop...")
    current_task = None
    while True:
        if current_task and not current_task.done():
            try:
                await asyncio.sleep(0.5) 
            except asyncio.CancelledError:
                print("[Voice Loop] Sleep cancelled, exiting loop.")
                if current_task and not current_task.done():
                    current_task.cancel() # Ensure inner task is also cancelled
                break # Exit loop on cancellation
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
                await asyncio.sleep(2) # Consider if this sleep should also be cancellable
            current_task = None # Clear task reference

        except asyncio.CancelledError:
            print("[Voice Loop] Interaction task cancelled, exiting loop.")
            if current_task and not current_task.done():
                 current_task.cancel()
            break
        
        except Exception as loop_e:
            print(f"\\nError during voice interaction loop iteration: {loop_e}")
            traceback.print_exc()
            current_task = None # Clear task on other exceptions too
            try:
                await asyncio.sleep(2) # Allow brief recovery pause, check cancellation
            except asyncio.CancelledError:
                 print("[Voice Loop] Recovery sleep cancelled, exiting loop.")
                 break