from typing import *
from transformers import AutoModelForSeq2SeqLM

def generate_translation(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95):
    """Generate translation using the specified model.

    Args:
        inputs (str): The input text to be translated.
        max_new_tokens (int, optional): The maximum number of new tokens to generate.
        do_sample (bool, optional): Whether to use sampling for generation.
        top_k (int, optional): The number of highest probability tokens to keep for sampling.
        top_p (float, optional): The cumulative probability threshold for sampling.

    Returns:
        str: The generated translation."""
    model = AutoModelForSeq2SeqLM.from_pretrained('my_awesome_opus_books_model')
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
    return outputs
