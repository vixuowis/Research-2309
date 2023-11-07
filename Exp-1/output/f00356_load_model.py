from typing import *
from transformers import TFAutoModelForSeq2SeqLM

def load_model(checkpoint):
	model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
	return model
