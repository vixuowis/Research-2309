from typing import *
from transformers import Trainer

def push_to_hub(self) -> str:
    """
    Pushes the final model to the 🤗 Hub.

    Returns:
        str: The URL of the pushed model.
    """
    return self.model.push_to_hub()
