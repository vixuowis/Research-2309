from typing import *
from transformers import AutoModelForCausalLM

def load_peft_adapter(model_id: str, peft_model_id: str) -> AutoModelForCausalLM:
    """Load a PEFT adapter for a given model.

    Args:
        model_id (str): The ID of the base model.
        peft_model_id (str): The ID of the PEFT adapter model.

    Returns:
        model (AutoModelForCausalLM): The model with the PEFT adapter loaded.
    """
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.load_adapter(peft_model_id)
    return model
