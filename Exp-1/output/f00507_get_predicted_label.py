from typing import *
import tensorflow as tf

def get_predicted_label(logits, id2label):
	# Get the predicted class id
	predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
	# Convert class id to label
	label = id2label[predicted_class_id]
	return label
