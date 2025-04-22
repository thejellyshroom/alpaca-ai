import asyncio
import traceback

async def run_voice_interaction_loop(assistant, duration, timeout, phrase_limit):
    """Runs the main interaction loop for voice mode."""
    print("Starting VOICE mode interaction loop...")
    while True: # Loop indefinitely until cancelled
        interaction_task = None
        try:
            # Create the interaction task
            interaction_task = asyncio.create_task(
                assistant.interaction_handler.run_single_interaction(
                    duration=duration,
                    timeout=timeout,
                    phrase_limit=phrase_limit
                ),
                name="SingleVoiceInteraction"
            )

            # Poll while the interaction task runs, checking for cancellation
            while not interaction_task.done():
                try:
                    await asyncio.sleep(0.1) # Yield control
                except asyncio.CancelledError:
                    print("[Voice Loop] Polling sleep cancelled. Cancelling interaction task...")
                    if not interaction_task.done():
                        interaction_task.cancel()
                    # Wait briefly for cancellation to propagate if needed
                    await asyncio.sleep(0.01)
                    raise # Re-raise CancelledError to exit the main voice loop

            # Interaction task is done, get the result or handle exceptions
            user_input_status, assistant_output = await interaction_task
            
            if user_input_status == "ERROR":
                print(f"Recovering from interaction error: {assistant_output}")
                try:
                     # Make recovery sleep cancellable too
                     await asyncio.sleep(2)
                except asyncio.CancelledError:
                    print("[Voice Loop] Recovery sleep cancelled, exiting loop.")
                    break # Exit loop

        except asyncio.CancelledError:
            print("[Voice Loop] Interaction task or loop cancelled, exiting.")
            # The main task (running this loop) was cancelled by the signal handler
            # Or the polling loop caught cancellation and re-raised it.
            if interaction_task and not interaction_task.done():
                 # Attempt to cancel the inner interaction task if it's still running
                 interaction_task.cancel()
            break # Exit the while loop
        
        except Exception as loop_e:
            print(f"\nError during voice interaction: {loop_e}")
            traceback.print_exc()
            # Ensure task is cancelled if an unexpected error occurred during processing
            if interaction_task and not interaction_task.done():
                 interaction_task.cancel()
            try:
                await asyncio.sleep(2) # Allow brief recovery pause, check cancellation
            except asyncio.CancelledError:
                 print("[Voice Loop] Recovery sleep cancelled, exiting loop.")
                 break # Exit loop 