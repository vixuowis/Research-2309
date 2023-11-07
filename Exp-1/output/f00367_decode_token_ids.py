from typing import *
from transformers import T5Tokenizer

def decode_token_ids(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)
