from typing import *
from transformers import AutoModelForCausalLM

def save_and_load_adapter(save_dir):
    """Save and load a trained adapter.

    Args:
        save_dir (str): The directory to save the adapter.

    Returns:
        model (AutoModelForCausalLM): The loaded model with the saved adapter.
    """
    model.save_pretrained(save_dir)
    model = AutoModelForCausalLM.from_pretrained(save_dir)
