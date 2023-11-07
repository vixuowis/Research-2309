from typing import *
from transformers import AutoModelForSequenceClassification

def load_model() -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
    return model
