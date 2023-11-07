from typing import *
from transformers import AutoModelForCausalLM

def load_model():
    """
    Load the model

    Returns:
        model: Pretrained LLM model
    """
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
    )
