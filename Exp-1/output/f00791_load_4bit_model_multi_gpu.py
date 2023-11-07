from typing import *
from transformers import AutoModelForCausalLM

def load_4bit_model_multi_gpu(model_name: str) -> AutoModelForCausalLM:
    """Load a mixed 4-bit model in multiple GPUs.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        AutoModelForCausalLM: The loaded model.
    """
    model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    return model_4bit
