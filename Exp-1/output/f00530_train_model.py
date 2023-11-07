from typing import *
import tensorflow as tf

def train_model(tf_train_dataset, tf_eval_dataset, callbacks, num_epochs):
    """
    Train the model.

    Args:
        tf_train_dataset: The training dataset.
        tf_eval_dataset: The validation dataset.
        callbacks: List of callbacks to be used during training.
        num_epochs: Number of epochs to train the model.

    Returns:
        None
    """
    model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset,
        callbacks=callbacks,
        epochs=num_epochs,
    )
