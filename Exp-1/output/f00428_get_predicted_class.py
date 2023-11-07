from typing import *
import tensorflow as tf

def get_predicted_class(logits):
	"""
	Get the class with the highest probability.

	Args:
		logits: A tensor representing the predicted logits.

	Returns:
		The predicted class as an integer.
	"""
	return int(tf.math.argmax(logits, axis=-1)[0])

