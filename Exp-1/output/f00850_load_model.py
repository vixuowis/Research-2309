from typing import *
from transformers import AutoModelForCausalLM

def load_model(model_name: str) -> AutoModelForCausalLM:
    """
    Load the language model from Hugging Face model hub.

    Args:
        - model_name (str): The name of the model to load.

    Returns:
        - AutoModelForCausalLM: The loaded language model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', pad_token_id=0)
    return model
