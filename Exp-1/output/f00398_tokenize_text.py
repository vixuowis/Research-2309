from typing import *
from transformers import AutoTokenizer

def tokenize_text(text):
	tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
	inputs = tokenizer(text, return_tensors="tf").input_ids
	return inputs
