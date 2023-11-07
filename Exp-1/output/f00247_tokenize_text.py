from typing import *
from transformers import AutoTokenizer

def tokenize_text(text):
	tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
	inputs = tokenizer(text, return_tensors="pt")
	return inputs
