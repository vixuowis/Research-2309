from typing import *
import tensorflow as tf

def get_highest_probability(outputs):
	'''Get the highest probability from the model output for the start and end positions:

	Args:
		outputs: model output containing start_logits and end_logits

	Returns:
		answer_start_index: highest probability index for start position
		answer_end_index: highest probability index for end position
	'''
	answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
	answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
	return answer_start_index, answer_end_index

