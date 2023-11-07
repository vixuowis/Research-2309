from typing import *
import torch

def generate_python_code(model, pixel_values):
	"""
	Pass your input to the model and return the `logits`:

	:param model: The model to generate code for
	:type model: torch.nn.Module
	:param pixel_values: The input pixel values
	:type pixel_values: torch.Tensor
	:return: The logits
	:rtype: torch.Tensor
	"""
	outputs = model(pixel_values=pixel_values)
	logits = outputs.logits.cpu()
	return logits

