from typing import *
from transformers import tokenizer

def tokenizer(text, return_tensors):
    pass



Tokenize the text:

    Args:
        text (str): The input text to be tokenized.
        return_tensors (str): The type of tensors to return (default: 'pt').

    Returns:
        encoded_text (Tensor): The encoded text.

def tokenizer(text, return_tensors):
    encoded_text = tokenizer(text, return_tensors)
    return encoded_text
