from typing import *
from transformers import AutoTokenizer

def preprocess(text: str) -> str:
    """
    Preprocesses the input text by tokenizing it using a DistilGPT2 tokenizer.
    
    Args:
        text (str): The input text to preprocess.
    
    Returns:
        str: The preprocessed text.
    """
    
    # Tokenize the input text
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
    
    # Convert the tokenized text back to a string
    preprocessed_text = tokenizer.decode(tokenized_text)
    
    return preprocessed_text
