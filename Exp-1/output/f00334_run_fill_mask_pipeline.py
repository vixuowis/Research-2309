from typing import *
from transformers import pipeline

def run_fill_mask_pipeline(text, model_name, top_k):
    mask_filler = pipeline("fill-mask", model_name)
    return mask_filler(text, top_k=top_k)
