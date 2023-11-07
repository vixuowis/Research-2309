from typing import *
import torch

def get_predicted_class(logits):
    predicted_class = logits.argmax().item()
    return predicted_class
