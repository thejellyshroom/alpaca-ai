import sys
import traceback
import types
import time # Import time

def run_text_interaction_loop(assistant):
    """Runs the main interaction loop for text mode synchronously."""
    print("Starting TEXT mode interaction loop... (Type 'quit' or 'exit' to stop)")
    
    while True:
        try:
            print("You: ", end="", flush=True)
            try:
                user_input_line = sys.stdin.readline()
                if not user_input_line: # Handle EOF
                     print("\nExiting text mode loop (EOF).")
                     break
                user_input = user_input_line.strip()
            except KeyboardInterrupt:
                 print("\n[Text Loop] KeyboardInterrupt received, exiting loop...")
                 break # Exit loop on Ctrl+C during input

            # Process input
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting text mode loop.")
                break

            if not user_input:
                continue

            print("Assistant: Thinking...") 
            response_generator = assistant.interaction_handler.run_single_text_interaction(user_input)

            # Stream the output and accumulate the full response
            response_chunks = []
            print("Assistant: ", end="", flush=True) # Print prefix before streaming
            try:
                if isinstance(response_generator, types.GeneratorType):
                    for chunk in response_generator:
                        print(chunk, end="", flush=True)
                        response_chunks.append(chunk)
                elif isinstance(response_generator, str):
                     print(response_generator)
                     response_chunks.append(response_generator)
                else:
                    print(f"\nError: Unexpected response type: {type(response_generator)}")
                    continue

                print() # Print final newline
            except Exception as e:
                print(f"\nError during response streaming: {e}")
                traceback.print_exc()
                continue

            full_response_text = "".join(response_chunks)
            if full_response_text and not full_response_text.startswith(("[Error", "ERROR:")):
                 assistant.conversation_manager.add_assistant_message(full_response_text)
            elif not full_response_text:
                 print("Warning: Assistant generated an empty response after streaming.")

        except KeyboardInterrupt:
            print("\n[Text Loop] KeyboardInterrupt caught, exiting.")
            break
        except Exception as loop_e:
            print(f"\nError during text interaction loop iteration: {loop_e}")
            traceback.print_exc()
            print("Attempting to recover...")
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                 print("\n[Text Loop] Recovery interrupted, exiting loop.")
                 break