from typing import *
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def generate_caption(model, tokenizer, prompt):
    """Generate a caption for an image using a few-shot prompting technique.

    Args:
        model (GPTNeoForCausalLM): The GPT-Neo model.
        tokenizer (GPT2Tokenizer): The GPT-2 tokenizer.
        prompt (list): A list of prompt strings.

    Returns:
        str: The generated caption.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    bad_words_ids = tokenizer(['<image>', '<fake_token_around_image>'], add_special_tokens=False).input_ids
    generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text[0]
