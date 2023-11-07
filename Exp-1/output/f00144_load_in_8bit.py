from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_in_8bit(peft_model_id):
    '''
    Load a model in 8-bit precision.
    
    Args:
        peft_model_id (str): The identifier of the model to load.
    
    Returns:
        model (AutoModelForCausalLM): The loaded model.
    '''
    model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", load_in_8bit=True)
