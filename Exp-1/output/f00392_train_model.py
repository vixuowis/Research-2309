from typing import *
import tensorflow as tf
from keras.models import Model

def train_model(model: Model, train_data: tf.data.Dataset, test_data: tf.data.Dataset, epochs: int, callbacks: list) -> None:
    """Trains the model using the given training and validation datasets.

    Args:
        model (Model): The Keras model to train.
        train_data (tf.data.Dataset): The training dataset.
        test_data (tf.data.Dataset): The validation dataset.
        epochs (int): The number of epochs to train the model.
        callbacks (list): List of Keras callbacks to use during training.
    """
    model.fit(x=train_data, validation_data=test_data, epochs=epochs, callbacks=callbacks)
