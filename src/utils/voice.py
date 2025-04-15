import torch
from pathlib import Path
import json
import os


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path) as f:
        return json.load(f)


def get_available_voices(voices_dir):
    voices_dir = Path(voices_dir)
    if not voices_dir.exists():
        return []
    return [f.stem for f in voices_dir.glob("*.pt")]


def validate_voice_name(voice_name, voices_dir):
    available_voices = get_available_voices(voices_dir)
    if voice_name not in available_voices:
        raise ValueError(
            f"Voice '{voice_name}' not found. Available voices: {', '.join(available_voices)}"
        )
    return True


def load_voice(voice_name, voices_dir):
    voices_dir = Path(voices_dir)
    assert voices_dir.exists(), f"Voices directory does not exist: {voices_dir}"
    assert voices_dir.is_dir(), f"Voices path is not a directory: {voices_dir}"

    validate_voice_name(voice_name, voices_dir)

    voice_path = voices_dir / f"{voice_name}.pt"
    assert voice_path.exists(), f"Voice file not found: {voice_path}"
    assert voice_path.is_file(), f"Voice path is not a file: {voice_path}"

    try:
        voice = torch.load(voice_path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Error loading voice file {voice_path}: {str(e)}")

    if not isinstance(voice, torch.Tensor):
        try:
            voice = torch.tensor(voice)
        except Exception as e:
            raise RuntimeError(f"Could not convert voice to tensor: {str(e)}")

    return voice


def quick_mix_voice(output_name, voices_dir, *voices, weights=None):
    """Mixes and saves voices with specified weights.

    Args:
        output_name (str): The name of the output mixed voice file (without extension).
        voices_dir (str): The path to the directory containing voice files.
        *voices (torch.Tensor): Variable number of voice tensors to mix.
        weights (list, optional): List of weights for each voice. Defaults to equal weights if None.

    Returns:
        torch.Tensor: The mixed voice as a torch tensor.

    Raises:
        ValueError: If no voices are provided, if the number of weights does not match the number of voices, or if the sum of weights is not positive.
        AssertionError: If the voices directory does not exist or is not a directory.
    """
    voices_dir = Path(voices_dir)
    assert voices_dir.exists(), f"Voices directory does not exist: {voices_dir}"
    assert voices_dir.is_dir(), f"Voices path is not a directory: {voices_dir}"

    if not voices:
        raise ValueError("Must provide at least one voice")

    base_shape = voices[0].shape
    for i, voice in enumerate(voices):
        if not isinstance(voice, torch.Tensor):
            raise ValueError(f"Voice {i} is not a tensor")
        if voice.shape != base_shape:
            raise ValueError(
                f"Voice {i} has shape {voice.shape}, but expected {base_shape} (same as first voice)"
            )

    if weights is None:
        weights = [1.0 / len(voices)] * len(voices)
    else:
        if len(weights) != len(voices):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of voices ({len(voices)})"
            )
        weights_sum = sum(weights)
        if weights_sum <= 0:
            raise ValueError("Sum of weights must be positive")
        weights = [w / weights_sum for w in weights]

    device = voices[0].device
    voices = [v.to(device) for v in voices]

    stacked = torch.stack(voices)
    weights = torch.tensor(weights, device=device)

    mixed = torch.zeros_like(voices[0])
    for i, weight in enumerate(weights):
        mixed += stacked[i] * weight

    output_path = voices_dir / f"{output_name}.pt"
    torch.save(mixed, output_path)
    print(f"Created mixed voice: {output_name}.pt")
    return mixed


def split_into_sentences(text):
    """Splits text into sentences using more robust rules.

    Args:
        text (str): The input text to split.

    Returns:
        list: A list of sentences (strings).
    """
    import re

    text = text.strip()
    if not text:
        return []

    abbreviations = {
        "Mr.": "Mr",
        "Mrs.": "Mrs",
        "Dr.": "Dr",
        "Ms.": "Ms",
        "Prof.": "Prof",
        "Sr.": "Sr",
        "Jr.": "Jr",
        "vs.": "vs",
        "etc.": "etc",
        "i.e.": "ie",
        "e.g.": "eg",
        "a.m.": "am",
        "p.m.": "pm",
    }

    for abbr, repl in abbreviations.items():
        text = text.replace(abbr, repl)

    sentences = []
    current = []

    words = re.findall(r"\S+|\s+", text)

    for word in words:
        current.append(word)

        if re.search(r"[.!?]+$", word):
            if not re.match(r"^[A-Z][a-z]{1,2}$", word[:-1]):
                sentence = "".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
                continue

    if current:
        sentence = "".join(current).strip()
        if sentence:
            sentences.append(sentence)

    for abbr, repl in abbreviations.items():
        sentences = [s.replace(repl, abbr) for s in sentences]

    sentences = [s.strip() for s in sentences if s.strip()]

    final_sentences = []
    for s in sentences:
        if len(s) > 200:
            parts = s.split(",")
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                final_sentences.extend(parts)
            else:
                final_sentences.append(s)
        else:
            final_sentences.append(s)

    return final_sentences
