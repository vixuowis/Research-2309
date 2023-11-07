from typing import *
import tensorflow as tf

def save_pretrained(model, path):
    """Save the TensorFlow model with its new checkpoint.

    Args:
        model (tf.keras.Model): The TensorFlow model to be saved.
        path (str): The path where the model and its checkpoint will be saved.

    Returns:
        None
    """
    model.save(path)
