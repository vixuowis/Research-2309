from typing import *
import bitsandbytes

def load_model(model_path, load_in_8bit=False):
    """Load a model from a given path.

    :param model_path: The path to the model file.
    :param load_in_8bit: Whether to load the model in 8-bit format.

    :return: The loaded model.
    """
    if load_in_8bit:
        model = bitsandbytes.load_model(model_path, 8)
    else:
        model = bitsandbytes.load_model(model_path)

    return model
