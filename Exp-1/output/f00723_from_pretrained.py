from typing import *
from transformers import ResnetConfig
import json

def from_pretrained(config_path: str) -> ResnetConfig:
    '''
    Load a configuration file from a file path and instantiate a ResnetConfig.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        ResnetConfig: The instantiated ResnetConfig object.
    '''
    config_dict = {}  # Placeholder for the loaded configuration

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = ResnetConfig(**config_dict)  # Instantiate a ResnetConfig object

    return config
