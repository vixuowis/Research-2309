from typing import *
from transformers import TFDistilBertModel

def from_pretrained(model_name_or_path, config=None, *inputs, **kwargs):
    """Loads a pretrained model from a given model_name_or_path, or the model_name_or_path of a pretrained model.

    Args:
        - model_name_or_path: str - The model_name_or_path of a pretrained model configuration to load from.
        - config: PretrainedConfig - An optional configuration object to be used instead of the default one.

    Returns:
        - TFDistilBertModel - The loaded pretrained model."""
    tf_model = TFDistilBertModel.from_pretrained(model_name_or_path, config=my_config)
