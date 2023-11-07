from typing import *
from transformers import TFAutoModelForSequenceClassification

def load_distilbert_model(num_labels, id2label, label2id):
	model = TFAutoModelForSequenceClassification.from_pretrained(
		"distilbert-base-uncased", num_labels=num_labels, id2label=id2label, label2id=label2id
	)
	return model
