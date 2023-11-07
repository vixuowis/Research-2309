from typing import *
from transformers import TFAutoModelForSequenceClassification

def train_model():
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
