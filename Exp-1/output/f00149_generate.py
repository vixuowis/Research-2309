from typing import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate(input_text):
    '''
    Generate text based on input text.

    Args:
        input_text (str): The input text to generate from.

    Returns:
        str: The generated text.
    '''
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
