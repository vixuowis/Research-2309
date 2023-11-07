from f00279_get_highest_probability import *
def test_get_highest_probability():
	# Test case 1
	outputs = {
		'start_logits': tf.constant([[0.1, 0.3, 0.5, 0.2]]),
		'end_logits': tf.constant([[0.2, 0.1, 0.4, 0.3]])
	}
	answer_start_index, answer_end_index = get_highest_probability(outputs)
	assert answer_start_index == 2
	assert answer_end_index == 2

	# Test case 2
	outputs = {
		'start_logits': tf.constant([[0.3, 0.2, 0.1, 0.4]]),
		'end_logits': tf.constant([[0.4, 0.3, 0.2, 0.1]])
	}
	answer_start_index, answer_end_index = get_highest_probability(outputs)
	assert answer_start_index == 3
	assert answer_end_index == 0

	# Test case 3
	outputs = {
		'start_logits': tf.constant([[0.5, 0.1, 0.2, 0.3]]),
		'end_logits': tf.constant([[0.1, 0.4, 0.3, 0.2]])
	}
	answer_start_index, answer_end_index = get_highest_probability(outputs)
	assert answer_start_index == 0
	assert answer_end_index == 1

	# Test case 4
	outputs = {
		'start_logits': tf.constant([[0.2, 0.4, 0.1, 0.3]]),
		'end_logits': tf.constant([[0.3, 0.1, 0.2, 0.4]])
	}
	answer_start_index, answer_end_index = get_highest_probability(outputs)
	assert answer_start_index == 1
	assert answer_end_index == 3

	# Test case 5
	outputs = {
		'start_logits': tf.constant([[0.4, 0.3, 0.2, 0.1]]),
		'end_logits': tf.constant([[0.2, 0.3, 0.1, 0.4]])
	}
	answer_start_index, answer_end_index = get_highest_probability(outputs)
	assert answer_start_index == 0
	assert answer_end_index == 3

print('All test cases pass')

test_get_highest_probability()
