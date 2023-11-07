from typing import *
from transformers import BertTokenizer

def subword_tokenization(text):
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	tokens = tokenizer.tokenize(text)
	return tokens
