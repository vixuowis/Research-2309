from typing import *
from transformers import AutoTokenizer

def preprocess(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
