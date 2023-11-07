from typing import *
from transformers import AutoTokenizer

def preprocess_text_input(text: str, model_name: str, padding_side: str) -> dict:
    """Preprocesses the text input using the specified tokenizer.

    Args:
        text (str): The text to preprocess.
        model_name (str): The name of the pretrained model.
        padding_side (str): The side to apply padding.

    Returns:
        dict: The preprocessed model inputs.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    return model_inputs
