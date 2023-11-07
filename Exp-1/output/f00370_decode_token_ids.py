from typing import *
from transformers import AutoTokenizer

def decode_token_ids(tokenizer, token_ids):
    '''
    Decodes the generated token ids back into text

    Args:
        tokenizer (AutoTokenizer): The tokenizer used to encode the text
        token_ids (list): The token ids to be decoded

    Returns:
        str: The decoded text
    '''
    return tokenizer.decode(token_ids, skip_special_tokens=True)
