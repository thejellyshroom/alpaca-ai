import traceback
import time # Import time for sleep

def run_voice_interaction_loop(assistant, duration, timeout, phrase_limit):
    """Runs the main interaction loop for voice mode synchronously."""
    print("Starting VOICE mode interaction loop...")
    while True:
        try:
            user_input_status, assistant_output = assistant.interaction_handler.run_single_interaction(
                duration=duration,
                timeout=timeout,
                phrase_limit=phrase_limit
            )
            
            if user_input_status == "ERROR":
                print(f"Recovering from interaction error: {assistant_output}")
                time.sleep(2)
            elif user_input_status == "INTERRUPTED":
                 print("Interaction interrupted, starting new loop.")

        except KeyboardInterrupt:
            print("[Voice Loop] KeyboardInterrupt received, propagating up...")
            raise
            
        except Exception as loop_e:
            print(f"\nUnexpected error during voice interaction: {loop_e}")
            traceback.print_exc()
            print("Attempting to recover after error...")
            time.sleep(2)
