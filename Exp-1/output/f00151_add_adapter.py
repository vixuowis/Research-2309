from typing import *
from transformers import AutoModel
from transformers.adapters import AdapterConfig

def add_adapter(model, adapter_name):
    adapter_config = AdapterConfig.load(adapter_name)
    model.add_adapter(adapter_config)
    
    return None
