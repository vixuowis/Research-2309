from typing import *
from datasets import Dataset

def preprocess_function(example: dict) -> dict:
    """Preprocesses each example in the dataset.

    Args:
        example (dict): The input example.

    Returns:
        dict: The preprocessed example."""
    text = example['text']

    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Update the example
    example['text'] = preprocessed_text

    return example
