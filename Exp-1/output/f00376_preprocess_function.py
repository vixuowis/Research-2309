from typing import *
from transformers import T5Tokenizer

def preprocess_function(examples):
    # Prefixes the input with a prompt
    prefix = "summarize: "

    # Tokenize inputs
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Tokenize labels
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    # Assign labels to model inputs
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
