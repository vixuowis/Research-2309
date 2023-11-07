from typing import *
import torch
from transformers import XLMTokenizer, XLMWithLMHeadModel

def load_xlm_model():
    # Load the xlm-clm-enfr-1024 checkpoint (Causal language modeling, English-French)
    tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
    model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
