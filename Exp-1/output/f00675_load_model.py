from typing import *
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model(checkpoint):
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    return model, tokenizer
