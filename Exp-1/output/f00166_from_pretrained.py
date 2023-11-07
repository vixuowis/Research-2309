from typing import *
from transformers import AutoModel

def from_pretrained(model_name_or_path, *model_args, **kwargs):
    """Loads a model from a pretrained model configuration or a directory.

    Args:
    - model_name_or_path: str - either:
        - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g., "bert-base-uncased".
        - a string with the `identifier name` of a pre-trained model configuration that was user-uploaded to the Hugging Face Hub, e.g., "dbmdz/bert-base-german-cased".
        - a path to a `directory` containing a configuration file saved using the `save_pretrained()` method, e.g., "./my_model_directory/".
    - model_args: tuple - (`optional`) Sequence of positional arguments:
        - All remaning positional arguments will be passed to the underlying model's `from_pretrained()` method.
    - kwargs: dict - (`optional`) Remaining dictionary of keyword arguments:
        - All remaining keyword arguments will be passed to the underlying model's `from_pretrained()` method.

    Returns:
        - A model instance of the specified pre-trained model configuration.
    """
    return AutoModel.from_pretrained(model_name_or_path, *model_args, **kwargs)
