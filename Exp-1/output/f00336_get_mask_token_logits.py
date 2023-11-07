from typing import *
from transformers import AutoModelForMaskedLM

def get_mask_token_logits(model, inputs, mask_token_index):
	"""
	Pass your inputs to the model and return the logits of the masked token:

	Args:
		model (AutoModelForMaskedLM): The pre-trained model for masked language modeling.
		inputs (dict): The inputs to the model.
		mask_token_index (int): The index of the masked token.

	Returns:
		mask_token_logits (torch.Tensor): The logits of the masked token.
	"""
	logits = model(**inputs).logits
	mask_token_logits = logits[0, mask_token_index, :]
	return mask_token_logits

