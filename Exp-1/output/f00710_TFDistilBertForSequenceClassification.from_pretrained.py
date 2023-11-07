from typing import *
from transformers import TFDistilBertForSequenceClassification

TFDistilBertForSequenceClassification is a base DistilBERT model with a sequence classification head. The sequence classification head is a linear layer on top of the pooled outputs.

tf_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
