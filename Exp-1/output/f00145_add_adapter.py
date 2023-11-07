from typing import *
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

def add_adapter(self, adapter_config: AdapterConfig, adapter_name: Optional[str] = None) -> str:
    # Generate a unique adapter name if not provided
    if adapter_name is None:
        adapter_name = self._generate_unique_adapter_name()
    
    # Check if the new adapter is the same type as the current adapter
    current_adapter_type = self.config.adapters.get(adapter_name, None)
    if current_adapter_type is None:
        raise ValueError(f'Adapter with name {adapter_name} does not exist in the model.')
    
    if adapter_config.adapter_type != current_adapter_type:
        raise ValueError(f'Adapter type {adapter_config.adapter_type} does not match the current adapter type {current_adapter_type}.')
    
    # Add the new adapter to the model
    self.config.adapters[adapter_name] = adapter_config
    
    return adapter_name
