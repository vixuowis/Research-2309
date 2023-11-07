from typing import *
from transformers import AutoModelForSequenceClassification

def load_model():
    # Load your model with the number of expected labels:
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
