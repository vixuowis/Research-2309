from typing import *
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

def generate_python_code(code_prompt):
    # Generate Python code based on the given prompt
    # Args:
    #     code_prompt (str): The prompt for generating Python code
    # Returns:
    #     generated_code (str): The generated Python code
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model_inputs = tokenizer(code_prompt, return_tensors='pt').to('cuda')
    generated_ids = model.generate(**model_inputs, max_new_tokens=200)
    generated_code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_code
