from typing import *
from transformers import AutoTokenizer

def tokenizer(sequence):
    # Tokenize the input sequence
    # Args:
    #   sequence (str): The input sequence to be tokenized
    # Returns:
    #   dict: A dictionary containing the tokenized input
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer.encode_plus(sequence, add_special_tokens=True, padding='longest', truncation=True, max_length=512, return_tensors='pt')
    return inputs

