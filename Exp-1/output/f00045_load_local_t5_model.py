from typing import *
from transformers import T5Model

def load_local_t5_model(model_path: str) -> T5Model:
    """
    Load a T5 model from a local directory.

    Args:
        model_path (str): The path to the local directory containing the model files.

    Returns:
        T5Model: The loaded T5 model.
    """
    model = T5Model.from_pretrained(model_path, local_files_only=True)
    return model
