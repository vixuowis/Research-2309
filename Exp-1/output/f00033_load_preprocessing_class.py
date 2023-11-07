from typing import *
from transformers import AutoTokenizer


def load_preprocessing_class(model_name: str) -> Any:
    """
    Load a preprocessing class like a tokenizer, image processor, feature extractor, or processor.

    Args:
        model_name (str): The name of the model to load the preprocessing class for.

    Returns:
        Any: The loaded preprocessing class.
    """
    preprocessing_class = AutoTokenizer.from_pretrained(model_name)
    return preprocessing_class
