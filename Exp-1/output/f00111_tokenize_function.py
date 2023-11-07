from typing import *
from transformers import AutoTokenizer

def tokenize_function(examples):
    """
    Preprocesses the examples by tokenizing the text using the tokenizer.

    Args:
        examples (dict): The input examples.

    Returns:
        dict: The tokenized examples.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)
