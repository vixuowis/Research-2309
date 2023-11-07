from typing import *
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

def load_blip2_model():
    """Load the BLIP-2 model for VQA

    Returns:
        model (Blip2ForConditionalGeneration): The BLIP-2 model for VQA
    """
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model
