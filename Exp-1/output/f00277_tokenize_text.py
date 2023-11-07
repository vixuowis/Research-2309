from typing import *
from transformers import AutoTokenizer

def tokenize_text(question, text):
	tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
	inputs = tokenizer(question, text, return_tensors="tf")

# Tokenize the text and return TensorFlow tensors:

# :param question: The question to be answered
# :param text: The text to search for the answer
# :return: TensorFlow tensors
