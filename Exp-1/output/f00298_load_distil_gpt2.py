from typing import *
from transformers import TFAutoModelForCausalLM

def load_distil_gpt2():
    """Load DistilGPT2 model.

    Returns:
        TFAutoModelForCausalLM: The loaded DistilGPT2 model."""
    model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")
    return model
