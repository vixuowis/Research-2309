from typing import *
from transformers import GPT2Tokenizer

def decode_token_ids(tokenizer, token_ids):
    '''
    Decode the generated token ids back into text
    
    Args:
        tokenizer (GPT2Tokenizer): The tokenizer used for encoding the text
        token_ids (List[int]): The token ids to decode
    
    Returns:
        List[str]: The decoded text
    '''
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)
