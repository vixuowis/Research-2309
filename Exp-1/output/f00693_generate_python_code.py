from typing import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_python_code(model, tokenizer, encoded_zh):
	generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
	return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
