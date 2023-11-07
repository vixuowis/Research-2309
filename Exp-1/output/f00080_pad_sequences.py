from typing import *
from transformers import AutoTokenizer
import torch

def pad_sequences(batch_sentences, padding=True):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    encoded_input = tokenizer(batch_sentences, padding=padding)
    return encoded_input
