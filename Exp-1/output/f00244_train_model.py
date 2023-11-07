from typing import *
import tensorflow as tf
from tensorflow import keras

def train_model(model, train_data, val_data, num_epochs, callbacks):
    '''
    Trains a model using the given training and validation datasets.

    Args:
        model (tf.keras.Model): The model to be trained.
        train_data (tf.data.Dataset): The training dataset.
        val_data (tf.data.Dataset): The validation dataset.
        num_epochs (int): The number of epochs to train the model.
        callbacks (list): List of callbacks to be used during training.
    '''
    model.fit(x=train_data, validation_data=val_data, epochs=num_epochs, callbacks=callbacks)
