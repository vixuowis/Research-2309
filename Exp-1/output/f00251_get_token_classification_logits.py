from typing import *
from transformers import TFAutoModelForTokenClassification

def get_token_classification_logits(model, inputs):
	logits = model(**inputs).logits
	return logits
