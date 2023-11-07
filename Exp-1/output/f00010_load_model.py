from typing import *
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def load_model(model_name):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
