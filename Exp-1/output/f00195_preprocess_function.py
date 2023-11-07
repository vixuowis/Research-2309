from typing import *
from transformers import DistilBertTokenizer

def preprocess_function(examples):
	return tokenizer(examples["text"], truncation=True)
