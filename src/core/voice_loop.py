import traceback
import time # Import time for sleep

def run_voice_interaction_loop(assistant, duration, timeout, phrase_limit):
    """Runs the main interaction loop for voice mode synchronously."""
    print("Starting VOICE mode interaction loop...")
    while True: # Loop indefinitely until KeyboardInterrupt
        try:
            # Directly call the synchronous interaction method
            user_input_status, assistant_output = assistant.interaction_handler.run_single_interaction(
                duration=duration,
                timeout=timeout,
                phrase_limit=phrase_limit
            )
            
            # Handle interaction status
            if user_input_status == "ERROR":
                print(f"Recovering from interaction error: {assistant_output}")
                time.sleep(2) # Simple synchronous sleep
            elif user_input_status == "INTERRUPTED":
                 print("Interaction interrupted, starting new loop.")
            # Other statuses (COMPLETED, TIMEOUT_ERROR, low_energy, etc.) just continue the loop

        except KeyboardInterrupt:
            # This loop doesn't handle KeyboardInterrupt directly;
            # it should be caught in the main function that calls this loop.
            print("[Voice Loop] KeyboardInterrupt received, propagating up...")
            raise # Re-raise to be caught by main
            
        except Exception as loop_e:
            print(f"\nUnexpected error during voice interaction: {loop_e}")
            traceback.print_exc()
            print("Attempting to recover after error...")
            time.sleep(2) # Recovery pause
            # Continue the loop after general errors 