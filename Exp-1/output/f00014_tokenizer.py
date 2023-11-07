from typing import *
from transformers import AutoTokenizer

def tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt'):
    
    pt_batch = tokenizer(texts, padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors)
