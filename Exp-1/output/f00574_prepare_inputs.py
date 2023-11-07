from typing import *
from transformers import CLIPTokenizer, CLIPProcessor
import torch

def prepare_inputs(text_queries, images):
    processor = CLIPProcessor()
    im = torch.tensor(images)
    inputs = processor(text=text_queries, images=im, return_tensors="pt")
    return inputs
