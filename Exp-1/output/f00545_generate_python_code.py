from typing import *
from transformers import Trainer
def generate_python_code(model_name: str, model_path: str) -> str:
    """Generate python code to push a trained model to the Hub.

    Args:
        model_name (str): The name of the model.
        model_path (str): The path to the trained model.

    Returns:
        str: The generated python code."""
    code = f'>>> trainer = Trainer.from_pretrained("{model_name}", "path/to/model")\n>>> trainer.push_to_hub()'
    return code
