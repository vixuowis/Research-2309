from typing import *
import torch


def load_model_with_exllama(username: str) -> torch.nn.Module:
    '''
    Load the GPTQ model with exllama kernels
    
    Args:
    - username (str): The username of the pretrained model
    
    Returns:
    - model (torch.nn.Module): The loaded GPTQ model
    '''
    gptq_config = GPTQConfig(bits=4, disable_exllama=False)
    model = AutoModelForCausalLM.from_pretrained(f'{username}/opt-125m-gptq', device_map='auto', quantization_config=gptq_config)
    return model
