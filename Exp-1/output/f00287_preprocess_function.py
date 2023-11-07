from typing import *
from transformers import tokenizer

def preprocess_function(examples):
    tokenized_examples = tokenizer([" ".join(x) for x in examples["answers.text"]])
    return tokenized_examples
