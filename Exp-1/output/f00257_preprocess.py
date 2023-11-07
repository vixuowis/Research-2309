from typing import *
from transformers import AutoTokenizer

def preprocess(question: str, context: str) -> dict:
    """Preprocesses the question and context using a DistilBERT tokenizer.
    
    Args:
        question (str): The input question.
        context (str): The input context.
    
    Returns:
        dict: A dictionary containing the preprocessed question and context."""
    encoded_input = tokenizer.encode_plus(question, context, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    preprocessed_data = {
        'input_ids': encoded_input['input_ids'].flatten(),
        'attention_mask': encoded_input['attention_mask'].flatten()
    }

    return preprocessed_data
