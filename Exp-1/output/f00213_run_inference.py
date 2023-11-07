from typing import *
from transformers import pipeline

def run_inference(text):
	"""
	Run inference on the given text using the finetuned model.

	Args:
		text (str): The text to run inference on.

	Returns:
		str: The predicted sentiment of the text.
	"""
	sentiment_pipeline = pipeline('text-classification', model='path/to/finetuned/model', tokenizer='path/to/tokenizer')
	sentiment = sentiment_pipeline(text)[0]['label']
	return sentiment
