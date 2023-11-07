from typing import *
from peft import PeftModel

def set_adapter(adapter: str) -> None:
    """Set the adapter to use for the model.

    Args:
        adapter (str): The name of the adapter to use.

    Returns:
        None
    """
    PeftModel.set_adapter(adapter)
