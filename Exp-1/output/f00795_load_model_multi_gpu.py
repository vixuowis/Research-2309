from typing import *
from transformers import AutoModelForCausalLM

def load_model_multi_gpu(model_name: str) -> AutoModelForCausalLM:
    """
    Load a mixed 8-bit model in multiple GPUs.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        AutoModelForCausalLM: The loaded model.
    """
    model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    return model_8bit
