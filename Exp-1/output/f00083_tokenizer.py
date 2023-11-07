from typing import *
from transformers import AutoTokenizer

def tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='tf'):
    '''
    Tokenizes a batch of sentences.

    Args:
        batch_sentences: A list of sentences to tokenize.
        padding: Whether to pad the sequences.
        truncation: Whether to truncate the sequences.
        return_tensors: The type of tensors to return (e.g., 'tf' for TensorFlow tensors).

    Returns:
        encoded_input: A dictionary containing the encoded input sequences.
    '''
    encoded_input = tokenizer(batch_sentences, padding=padding, truncation=truncation, return_tensors=return_tensors)
    return encoded_input
