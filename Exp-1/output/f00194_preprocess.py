from typing import *
from transformers import AutoTokenizer

def preprocess(text):
    """Preprocesses the text using a DistilBERT tokenizer.

    Args:
        text (str): The input text to preprocess.

    Returns:
        list: The preprocessed tokens.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return tokens

