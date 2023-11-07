from typing import *
from transformers import PreTrainedModel

def save_model(model, tokenizer, save_directory):
	"""Save a model with its tokenizer.

	Args:
		model (PreTrainedModel): The model to save.
		tokenizer (PreTrainedTokenizer): The tokenizer to save.
		save_directory (str): The directory to save the model and tokenizer in.

	Returns:
		None"""

	tokenizer.save_pretrained(save_directory)
	model.save_pretrained(save_directory)
