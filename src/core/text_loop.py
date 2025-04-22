import asyncio
import sys
import traceback
import types

async def run_text_interaction_loop(assistant):
    """Runs the main interaction loop for text mode."""
    print("Starting TEXT mode interaction loop... (Type 'quit' or 'exit' to stop)")
    
    while True:
        input_task = None
        try:
            print("You: ", end="", flush=True)
            
            input_task = asyncio.create_task(asyncio.to_thread(sys.stdin.readline))

            while not input_task.done():
                try:
                    # Sleep briefly to yield control and allow cancellation checks
                    await asyncio.sleep(0.1) 
                except asyncio.CancelledError:
                    print("[Text Loop] Sleep cancelled during input wait. Cancelling input task...")
                    if not input_task.done():
                        input_task.cancel()
                    await asyncio.sleep(0.01) 
                    raise

            if not input_task.done():
                print("[Text Loop] Error: Input task loop exited but task not done?")
                continue

            # Get the result
            try:
                user_input_line = input_task.result()
                user_input = user_input_line.strip()
            except asyncio.CancelledError:
                print("[Text Loop] Input task was cancelled before result retrieval.")
                raise
            except EOFError:
                 print("\nExiting text mode loop (EOF).")
                 break

            # Process input
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting text mode loop.")
                break

            if not user_input:
                continue

            # Call the text interaction handler
            print("Assistant: Thinking...") 

            # Get the response generator
            response_generator = await assistant.interaction_handler.run_single_text_interaction(user_input)

            # Stream the output and accumulate the full response
            response_chunks = []
            print("Assistant: ", end="", flush=True) # Print prefix before streaming
            try:
                # Check if the generator is async or sync
                if isinstance(response_generator, types.AsyncGeneratorType):
                    async for chunk in response_generator:
                        print(chunk, end="", flush=True)
                        response_chunks.append(chunk)
                elif isinstance(response_generator, types.GeneratorType):
                    for chunk in response_generator:
                        print(chunk, end="", flush=True)
                        response_chunks.append(chunk)
                        # Add a small sleep to yield control, might help responsiveness slightly
                        await asyncio.sleep(0.001)
                else:
                    print(f"\nError: Unexpected response generator type: {type(response_generator)}")
                    continue

                print() # Print final newline
            except Exception as e:
                print(f"\nError during response streaming: {e}")
                traceback.print_exc()
                continue

            # Join chunks and add full response to history
            full_response_text = "".join(response_chunks)
            if full_response_text and not full_response_text.startswith("[Error"):
                 assistant.conversation_manager.add_assistant_message(full_response_text)
            elif not full_response_text:
                 print("Warning: Assistant generated an empty response after streaming.")

        except asyncio.CancelledError:
            print("[Text Loop] Main task cancelled, exiting loop.")
            if input_task and not input_task.done():
                input_task.cancel()
            break
        except Exception as loop_e:
            print(f"\nError during text interaction loop iteration: {loop_e}")
            traceback.print_exc()
            if input_task and not input_task.done():
                input_task.cancel()
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                 print("[Text Loop] Recovery sleep cancelled, exiting loop.")
                 break