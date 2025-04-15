import torch
from datetime import datetime
from pathlib import Path
from .voice import quick_mix_voice


def handle_commands(user_input, generator, speed, model_path=None):
    """
    Handles bot commands to control the voice generator.

    Args:
        user_input (str): The command input from the user.
        generator: The voice generator object.
        speed (float): The current speed of the generator.
        model_path (str, optional): The path to the model. Defaults to None.

    Returns:
        bool: True if a command was handled, False otherwise.
    """
    if user_input.lower() == "quit":
        print("Goodbye!")
        return True

    if user_input.lower() == "voices":
        voices = generator.list_available_voices()
        print("\nAvailable voices:")
        for voice in voices:
            print(f"- {voice}")
        return True

    if user_input.startswith("speed="):
        try:
            new_speed = float(user_input.split("=")[1])
            print(f"Speed set to {new_speed}")
            return True
        except:
            print("Invalid speed value. Use format: speed=1.2")
            return True

    if user_input.startswith("voice="):
        try:
            voice = user_input.split("=")[1]
            if voice in generator.list_available_voices():
                generator.initialize(model_path or generator.model_path, voice)
                print(f"Switched to voice: {voice}")
            else:
                print("Voice not found. Use 'voices' to list available voices.")
        except Exception as e:
            print(f"Error changing voice: {str(e)}")
        return True

    if user_input.startswith("mix="):
        try:
            mix_input = user_input.split("=")[1]
            voices_weights = mix_input.split(":")
            voices = [v.strip() for v in voices_weights[0].split(",")]

            if len(voices_weights) > 1:
                weights = [float(w.strip()) for w in voices_weights[1].split(",")]
            else:
                weights = [0.5, 0.5]

            if len(voices) != 2 or len(weights) != 2:
                print(
                    "Mix command requires exactly two voices. Format: mix=voice1,voice2[:weight1,weight2]"
                )
                return True

            available_voices = generator.list_available_voices()
            if not all(voice in available_voices for voice in voices):
                print(
                    "One or more voices not found. Use 'voices' to list available voices."
                )
                return True

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"af_mixed_voice_{timestamp}"

            voice_tensors = []
            for voice_name in voices:
                voice_path = Path(generator.voices_dir) / f"{voice_name}.pt"
                voice = torch.load(voice_path, weights_only=True)
                voice_tensors.append(voice)

            mixed = quick_mix_voice(
                output_name, generator.voices_dir, *voice_tensors, weights=weights
            )

            generator.initialize(model_path or generator.model_path, output_name)
            print(
                f"Mixed voices: {voices[0]} ({weights[0]:.1f}) and {voices[1]} ({weights[1]:.1f})"
            )
        except Exception as e:
            print(f"Error mixing voices: {str(e)}")
        return True

    return False
