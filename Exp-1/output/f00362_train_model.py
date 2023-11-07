from typing import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model

def train_model(model: Model, train_set, test_set, epochs: int, callbacks: Callback) -> None:
    """Train the model using the given training and validation datasets.

    Args:
        model (Model): The Keras model to be trained.
        train_set: The training dataset.
        test_set: The validation dataset.
        epochs (int): The number of epochs to train the model.
        callbacks (Callback): The callbacks to be used during training.
    """
    model.fit(x=train_set, validation_data=test_set, epochs=epochs, callbacks=callbacks)
