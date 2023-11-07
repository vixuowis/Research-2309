from typing import *
from transformers import pipeline

def pipeline(model: str, return_timestamps: bool) -> Callable:
    """Performs automatic speech recognition on audio files.

    Args:
        model (str): The pre-trained model to use for speech recognition.
        return_timestamps (bool): Whether to return timestamps for each recognized chunk.

    Returns:
        Callable: A function that accepts an audio file URL as input and returns the transcriptions."""
    transcriber = pipeline(model=model, return_timestamps=return_timestamps)
