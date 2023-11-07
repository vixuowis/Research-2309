from typing import *
from transformers import AutoModelForCausalLM
from accelerate import Accelerator

def load_model_with_memory(model_name: str, device_map: str, load_in_4bit: bool, max_memory_mapping: dict) -> AutoModelForCausalLM:
    """
    Load a model with specified GPU memory allocation.
    
    Args:
    - model_name (str): The name of the model to load.
    - device_map (str): The device mapping strategy.
    - load_in_4bit (bool): Whether to load the model in 4-bit mode.
    - max_memory_mapping (dict): A dictionary mapping GPU IDs to maximum memory allocations.
    
    Returns:
    - AutoModelForCausalLM: The loaded model.
    """
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, load_in_4bit=load_in_4bit, max_memory=max_memory_mapping
    )
    return model
