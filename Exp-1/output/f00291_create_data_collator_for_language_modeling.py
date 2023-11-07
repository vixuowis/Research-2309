from typing import *
from transformers import DataCollatorForLanguageModeling

def create_data_collator_for_language_modeling(tokenizer):
	tokenizer.pad_token = tokenizer.eos_token
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
	return data_collator
