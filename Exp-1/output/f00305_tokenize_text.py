from typing import *
from transformers import AutoTokenizer

def tokenize_text(text, model_name):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	inputs = tokenizer(text, return_tensors='pt').input_ids
	return inputs
