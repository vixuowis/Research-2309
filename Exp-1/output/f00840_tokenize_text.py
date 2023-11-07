from typing import *
from transformers import XLNetTokenizer

def tokenize_text(text):
	tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
	tokens = tokenizer.tokenize(text)
	return tokens
