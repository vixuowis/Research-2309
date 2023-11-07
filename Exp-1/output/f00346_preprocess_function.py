from typing import *
from transformers import AutoTokenizer

def preprocess_function(examples):
	# Prefix the input with a prompt
	# Tokenize the input (English) and target (French) separately
	# Truncate sequences to be no longer than the maximum length set by the `max_length` parameter.
	inputs = [prefix + example[source_lang] for example in examples["translation"]]
	targets = [example[target_lang] for example in examples["translation"]]
	model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
	return model_inputs
