from typing import *
import tensorflow as tf

def get_predicted_class(logits, id2label):
	"""Get the class with the highest probability and convert it to a text label.

	Args:
		logits: A tensor containing the predicted probabilities for each class.
		id2label: A dictionary mapping class IDs to text labels.

	Returns:
		The text label corresponding to the class with the highest probability.
	"""
	predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
	return id2label[predicted_class_id]
