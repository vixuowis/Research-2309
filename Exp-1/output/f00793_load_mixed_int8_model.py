from typing import *
from transformers import AutoModelForCausalLM

def load_mixed_int8_model(model_name: str) -> AutoModelForCausalLM:
    """Load a mixed 8-bit model for single GPU setup.

    Args:
        - model_name (str): The name of the model to load.

    Returns:
        - AutoModelForCausalLM: The loaded mixed 8-bit model.
    """
    model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    return model_8bit
