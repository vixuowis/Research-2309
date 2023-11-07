from typing import *
from transformers import AutoTokenizer


def preprocess(text: str, summary: str) -> Tuple[List[int], List[int]]:
    """
    Preprocesses the input text and summary using a T5 tokenizer.

    Args:
        text (str): The input text.
        summary (str): The input summary.

    Returns:
        Tuple[List[int], List[int]]: The processed input text and summary as token IDs.
    """
    input_ids = tokenizer.encode(text, truncation=True, padding='longest')
    summary_ids = tokenizer.encode(summary, truncation=True, padding='longest')

    return input_ids, summary_ids
