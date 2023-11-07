from typing import *
from transformers import AutoTokenizer

def tokenize_text(text: str) -> dict:
    # Tokenize the text and return PyTorch tensors
    tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
    inputs = tokenizer(text, return_tensors="pt")
