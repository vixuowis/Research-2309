from typing import *
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(input_ids):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Generate text using the GPT-2 model.
    # Args:
    #     input_ids (torch.Tensor): The input tensor containing the tokenized text.
    # Returns:
    #     str: The generated text.
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
