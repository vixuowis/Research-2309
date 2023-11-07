from typing import *
import tensorflow as tf

def compile(optimizer):
    """Configure the model for training with `compile`.

    Args:
        - optimizer: The optimizer to use for training the model.

    Returns:
        The compiled model.
    """
    model.compile(optimizer=optimizer)
