from typing import *
from transformers import AutoTokenizer

def tokenize_text(text):
    """Tokenize the text and return the `input_ids` as PyTorch tensors:

    Args:
        text (str): The input text to tokenize.

    Returns:
        torch.Tensor: The tokenized input as PyTorch tensor."""
    tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
    inputs = tokenizer(text, return_tensors="pt").input_ids
