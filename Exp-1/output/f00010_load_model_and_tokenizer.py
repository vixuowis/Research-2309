from typing import *
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def load_model_and_tokenizer(model_name: str):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
