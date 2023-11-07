from typing import *
from transformers.keras_callbacks import PushToHubCallback

def push_to_hub_callback(output_dir: str, tokenizer: Any) -> PushToHubCallback:
    """This function creates a callback to push the model and tokenizer to the Hugging Face Hub.

    Args:
        output_dir (str): The directory where the model will be saved.
        tokenizer (Any): The tokenizer used for the model.

    Returns:
        PushToHubCallback: The callback object."""
    callback = PushToHubCallback(output_dir=output_dir, tokenizer=tokenizer)
