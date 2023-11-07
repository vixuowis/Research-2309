from typing import *
from transformers import TFPreTrainedModel
from tensorflow import keras

def load_tf_model(model_path: str) -> TFPreTrainedModel:
    """
    Load a TensorFlow model from the given path.

    Args:
        model_path (str): The path to the saved model.

    Returns:
        TFPreTrainedModel: The loaded TensorFlow model.
    """
    model = TFPreTrainedModel.from_pretrained(model_path)
    return model
