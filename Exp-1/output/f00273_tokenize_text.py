from typing import *
from transformers import AutoTokenizer

def tokenize_text(question, context):
	tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
	inputs = tokenizer(question, context, return_tensors="pt")

# Tokenize the text and return PyTorch tensors:

# Parameters:
# 	question (str): The question text.
# 	context (str): The context text.

# Returns:
# 	inputs (dict): A dictionary containing the tokenized inputs as PyTorch tensors.
