from typing import *
from transformers import AutoTokenizer

def tokenize_text(text):
	"""
	Tokenize the text and return the `input_ids` as PyTorch tensors:

	Args:
		text (str): The input text to tokenize.

	Returns:
		torch.Tensor: The input text tokenized as PyTorch tensors.
	"""
	tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
	inputs = tokenizer(text, return_tensors="pt").input_ids
	return inputs
