from typing import *
from transformers import Trainer

def push_to_hub():
    """Pushes the trained model to the Hub.

    Returns:
        str: The URL of the pushed model on the Hub."""
    return trainer.push_to_hub()
