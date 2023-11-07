from typing import *
from transformers import AutoTokenizer
from typing import List

def preprocess_text(text: str, tokenizer: AutoTokenizer) -> List[str]:
    '''Preprocesses a text by tokenizing it using a given tokenizer.

    Args:
        text (str): The input text to preprocess.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.

    Returns:
        List[str]: The list of tokens after tokenization.
    '''
    tokens = tokenizer.tokenize(text)
    return tokens
