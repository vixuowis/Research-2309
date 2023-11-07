from typing import *
from transformers import AutoTokenizer
import torch

def tokenize_text(text):
	tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_eli5_mlm_model")
	inputs = tokenizer(text, return_tensors="pt")
	mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
	return inputs, mask_token_index
