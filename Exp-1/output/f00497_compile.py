from typing import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def compile(optimizer, loss):
    """Configure the model for training with `compile()`:

    Args:
        - optimizer: The optimizer to use for training.
        - loss: The loss function to use for training.

    Returns:
        - None
    """
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)
