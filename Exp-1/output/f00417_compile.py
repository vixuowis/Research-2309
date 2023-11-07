from typing import *
from tensorflow.keras import Model

def compile(self, optimizer):
    """Configure the model for training with compile method.

    Args:
        optimizer (str or tf.keras.optimizers.Optimizer): The optimizer to use for training.

    Returns:
        None
    """
    self.model.compile(optimizer=optimizer)
