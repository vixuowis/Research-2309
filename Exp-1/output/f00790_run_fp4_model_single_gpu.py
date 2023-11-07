from typing import *
from transformers import AutoModelForCausalLM

def run_fp4_model_single_gpu(model_name: str) -> AutoModelForCausalLM:
    """
    Run a FP4 model on a single GPU.

    Args:
    - model_name (str): The name of the model to load.

    Returns:
    - AutoModelForCausalLM: The loaded FP4 model.
    """
    model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_4bit=True)
    return model_4bit
