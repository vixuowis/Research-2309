from typing import *
from transformers import AutoTokenizer

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
