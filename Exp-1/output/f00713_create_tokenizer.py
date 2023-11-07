from typing import *
from transformers import DistilBertTokenizer

def create_tokenizer(model_name: str) -> DistilBertTokenizer:
    """Create a tokenizer with a pretrained model's vocabulary.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        DistilBertTokenizer: The tokenizer with the pretrained model's vocabulary.
    """
    return DistilBertTokenizer.from_pretrained(model_name)
