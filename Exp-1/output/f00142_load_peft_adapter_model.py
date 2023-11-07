from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_peft_adapter_model(peft_model_id: str) -> AutoModelForCausalLM:
	"""
	Load a PEFT adapter model for causal language modeling.

	Args:
	- peft_model_id (str): The PEFT model id.

	Returns:
	- model (AutoModelForCausalLM): The loaded PEFT adapter model.
	"""
	model = AutoModelForCausalLM.from_pretrained(peft_model_id)
	return model
