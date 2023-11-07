from typing import *
from transformers import pipeline

def generate_audio(text: str) -> bytes:
    """
    Generate audio from text using the text-to-speech pipeline.

    Args:
        text (str): The input text to convert to audio.

    Returns:
        bytes: The audio data as bytes.
    """
    pipe = pipeline("text-to-speech", model="suno/bark-small")
    output = pipe(text)
    return output[0]["audio"]
