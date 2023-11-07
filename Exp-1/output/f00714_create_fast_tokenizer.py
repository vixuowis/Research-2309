from typing import *
from transformers import DistilBertTokenizerFast

def create_fast_tokenizer():
    """Create a fast tokenizer with the DistilBertTokenizerFast class:

    Returns:
        fast_tokenizer (DistilBertTokenizerFast): The created fast tokenizer
    """
    fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return fast_tokenizer
