from typing import *
import tensorflow as tf

def fit_model(model, train_set, test_set, epochs, callback):
	"""Fits the model to the training data and validates on the test data.

	Args:
	- model: The model to train.
	- train_set: The training dataset.
	- test_set: The test dataset.
	- epochs: The number of epochs to train for.
	- callback: The callback to use for fine-tuning the model.

	Returns:
	- None
	"""
	model.fit(x=train_set, validation_data=test_set, epochs=epochs, callbacks=[callback])
