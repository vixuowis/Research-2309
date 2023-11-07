from typing import *
from transformers import TFAutoModelForSeq2SeqLM

def generate_translation(inputs, model_path, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95):
    """Generate translation using the specified model and input text.

    Args:
        inputs (str): The input text to be translated.
        model_path (str): The path to the pre-trained model.
        max_new_tokens (int, optional): The maximum number of new tokens to generate.
        do_sample (bool, optional): Whether to use sampling during generation.
        top_k (int, optional): The number of highest probability tokens to consider for sampling.
        top_p (float, optional): The cumulative probability for sampling from the smallest possible set of tokens.

    Returns:
        str: The generated translation.
    """
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
    translation = outputs[0]['generated_text']
    return translation

