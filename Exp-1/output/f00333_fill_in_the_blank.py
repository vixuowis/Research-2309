from typing import *
from transformers import pipeline

def fill_in_the_blank(text: str) -> str:
    """
    Fill in the blank in the given text using a finetuned model.

    Args:
        text (str): The text with a blank to be filled in.

    Returns:
        str: The text with the blank filled in.
    """
    filler = pipeline('text-fill-mask')
    result = filler(text)
    filled_text = result[0]['sequence']
    return filled_text
