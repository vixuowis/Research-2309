from typing import *
import tensorflow as tf

def fit(x, validation_data, epochs, callbacks):
    """Fits the model on the training data and evaluates on the validation data.

    Args:
        - x: Training data.
        - validation_data: Validation data.
        - epochs: Number of epochs to train the model.
        - callbacks: List of callbacks to apply during training.

    Returns:
        - History object.
    """
    model.fit(x=x, validation_data=validation_data, epochs=epochs, callbacks=callbacks)
