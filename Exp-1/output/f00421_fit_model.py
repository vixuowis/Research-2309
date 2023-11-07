from typing import *
import tensorflow as tf

def fit_model(model, train_data, validation_data, num_epochs, callbacks):
    """Fits the model to the training data and validates it on the validation data.

    Args:
        model (tf.keras.Model): The model to be trained.
        train_data (tf.data.Dataset): The training dataset.
        validation_data (tf.data.Dataset): The validation dataset.
        num_epochs (int): The number of epochs to train the model.
        callbacks (list): List of callbacks to be applied during training.

    Returns:
        None
    """
    model.fit(x=train_data, validation_data=validation_data, epochs=num_epochs, callbacks=callbacks)
