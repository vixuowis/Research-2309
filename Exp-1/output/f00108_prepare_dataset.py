from typing import *
from transformers import Wav2Vec2Processor

def prepare_dataset(example):
    """This function processes the audio data contained in `array` to `input_values`, and tokenizes `text` to `labels`. These are the inputs to the model.

    Args:
        example (dict): A dictionary containing the audio data and text.

    Returns:
        dict: The updated example dictionary with the processed audio data and tokenized text."""
    audio = example["audio"]

    example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

    return example
