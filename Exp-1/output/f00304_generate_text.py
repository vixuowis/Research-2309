from typing import *
from transformers import pipeline

def generate_text(model_name, prompt):
	generator = pipeline("text-generation", model=model_name)
	output = generator(prompt)
	return output[0]['generated_text']
