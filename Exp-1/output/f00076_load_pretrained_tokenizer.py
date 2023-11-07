from typing import *
from transformers import AutoTokenizer

def load_pretrained_tokenizer(model_name: str) -> AutoTokenizer:
    '''
    Load a pretrained tokenizer.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        AutoTokenizer: The pretrained tokenizer.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
