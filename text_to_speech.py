import traceback
import time
import requests
import signal
import sys
from src.utils.config import settings
from src.utils import (
    VoiceGenerator,
    get_ai_response,
    play_audio_with_interrupt,
)
from src.utils.audio_queue import AudioGenerationQueue
from src.utils.llm import parse_stream_chunk
import threading
from src.utils.text_chunker import TextChunker


def signal_handler(sig, frame):
    print("\nStopping...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

settings.setup_directories()
timing_info = {
    "input_received": None,
    "llm_first_token": None,
    "audio_queued": None,
    "first_audio_play": None,
    "end": None,
}


def process_input(
    session: requests.Session,
    user_input: str,
    messages: list,
    generator: VoiceGenerator,
    speed: float,
) -> None:
    """Processes user input, generates a response, and handles audio output.

    Args:
        session (requests.Session): The requests session to use.
        user_input (str): The user's input text.
        messages (list): The list of messages to send to the LLM.
        generator (VoiceGenerator): The voice generator object.
        speed (float): The playback speed.
    """
    global timing_info
    timing_info = {k: None for k in timing_info}
    timing_info["input_received"] = time.perf_counter()

    messages.append({"role": "user", "content": user_input})
    print("\nThinking...")

    try:
        response_stream = get_ai_response(
            session=session,
            messages=messages,
            llm_model=settings.LLM_MODEL,
            llm_url=settings.OLLAMA_URL,
            max_tokens=settings.MAX_TOKENS,
            stream=True,
        )

        if not response_stream:
            print("Failed to get AI response stream.")
            return

        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()
        chunker = TextChunker()
        complete_response = []

        playback_thread = threading.Thread(
            target=lambda: audio_playback_worker(audio_queue)
        )
        playback_thread.daemon = True
        playback_thread.start()

        for chunk in response_stream:
            data = parse_stream_chunk(chunk)
            if not data or "choices" not in data:
                continue

            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                content = choice["delta"]["content"]
                if content:
                    if not timing_info["llm_first_token"]:
                        timing_info["llm_first_token"] = time.perf_counter()
                    print(content, end="", flush=True)
                    chunker.current_text.append(content)

                    text = "".join(chunker.current_text)
                    if chunker.should_process(text):
                        if not timing_info["audio_queued"]:
                            timing_info["audio_queued"] = time.perf_counter()
                        remaining = chunker.process(text, audio_queue)
                        chunker.current_text = [remaining]
                        complete_response.append(text[: len(text) - len(remaining)])

            if choice.get("finish_reason") == "stop":
                final_text = "".join(chunker.current_text).strip()
                if final_text:
                    chunker.process(final_text, audio_queue)
                    complete_response.append(final_text)
                break

        messages.append({"role": "assistant", "content": " ".join(complete_response)})
        print()

        time.sleep(0.1)
        audio_queue.stop()
        playback_thread.join()

        def playback_wrapper():
            timing_info["playback_start"] = time.perf_counter()
            audio_playback_worker(audio_queue)

        playback_thread = threading.Thread(target=playback_wrapper)
        playback_thread.start()
        playback_thread.join()

        timing_info["end"] = time.perf_counter()
        print_timing_chart(timing_info)

    except Exception as e:
        print(f"\nError during streaming: {str(e)}")
        if "audio_queue" in locals():
            audio_queue.stop()


def audio_playback_worker(audio_queue) -> None:
    """Manages audio playback in a separate thread.

    Args:
        audio_queue (AudioGenerationQueue): The audio queue object.
    """
    global timing_info

    try:
        while True:
            audio_data, _ = audio_queue.get_next_audio()
            if audio_data is not None:
                if not timing_info["first_audio_play"]:
                    timing_info["first_audio_play"] = time.perf_counter()

                play_audio_with_interrupt(audio_data)
            else:
                time.sleep(settings.PLAYBACK_DELAY)

            if (
                not audio_queue.is_running
                and audio_queue.sentence_queue.empty()
                and audio_queue.audio_queue.empty()
            ):
                break

    except Exception as e:
        print(f"Error in audio playback: {str(e)}")


def print_timing_chart(metrics):
    """Prints timing chart from global metrics"""
    base_time = metrics["input_received"]
    events = [
        ("Input received", metrics["input_received"]),
        ("LLM first token", metrics["llm_first_token"]),
        ("Audio queued", metrics["audio_queued"]),
        ("First audio played", metrics["first_audio_play"]),
        ("End-to-end response", metrics["end"]),
    ]

    print("\nTiming Chart:")
    print(f"{'Event':<25} | {'Time (s)':>9} | {'Δ+':>6}")
    print("-" * 45)

    prev_time = base_time
    for name, t in events:
        if t is None:
            continue
        elapsed = t - base_time
        delta = t - prev_time
        print(f"{name:<25} | {elapsed:9.2f} | {delta:6.2f}")
        prev_time = t


def main():
    """Main function to run the text-to-speech chat bot."""
    with requests.Session() as session:
        try:
            session = requests.Session()
            generator = VoiceGenerator(settings.VOICES_DIR)
            print("VoiceGenerator initialized.")

            init_message = generator.initialize(settings.VOICE_NAME, settings.SPEED)
            print(init_message)

            messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]

            print("\n=== Text-to-Speech Chat Bot Initializing ===")
            print("Device being used:", generator.device)
            print("\nInitializing voice generator...")
            result = generator.initialize(settings.TTS_MODEL, settings.VOICE_NAME)
            print(result)
            speed = settings.SPEED

            try:
                print("\nWarming up the LLM model...")
                health = session.get("http://localhost:11434", timeout=3)
                if health.status_code != 200:
                    print("Ollama not running! Start it first.")
                    return
                response_stream = get_ai_response(
                    session=session,
                    messages=[
                        {"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": "Hi!"},
                    ],
                    llm_model=settings.LLM_MODEL,
                    llm_url=settings.OLLAMA_URL,
                    max_tokens=settings.MAX_TOKENS,
                    stream=False,
                )
                if not response_stream:
                    print("Failed to initialize the AI model!")
                    return
            except requests.RequestException as e:
                print(f"Warmup failed: {str(e)}")

            print("\n=== Text-to-Speech Chat Bot Ready ===")
            print("Type your messages and press Enter to chat.")
            print("Type 'quit' to exit.")

            while True:
                try:
                    user_input = input("\nYou: ").strip()

                    if user_input.lower() == "quit":
                        print("Goodbye!")
                        break

                    if user_input:
                        process_input(session, user_input, messages, generator, speed)

                    if session is not None:
                        session.headers.update({"Connection": "keep-alive"})
                        if hasattr(session, "connection_pool"):
                            session.connection_pool.clear()

                except EOFError:
                    print("\nStopping...")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
