from typing import *
from tokenizer_module import tokenizer

def tokenize_dataset(dataset):
    """Tokenizes the dataset.

    Args:
    - dataset: The dataset to be tokenized.

    Returns:
    The tokenized dataset.
    """
    return tokenizer(dataset["text"])
