from f00220_get_predicted_class import *
def test_get_predicted_class():
	logits = tf.constant([[0.1, 0.9, 0.2]])
	id2label = {0: 'NEGATIVE', 1: 'POSITIVE', 2: 'NEUTRAL'}
	assert get_predicted_class(logits, id2label) == 'POSITIVE'

	logits = tf.constant([[0.3, 0.2, 0.5]])
	assert get_predicted_class(logits, id2label) == 'NEUTRAL'

	logits = tf.constant([[0.7, 0.1, 0.2]])
	assert get_predicted_class(logits, id2label) == 'NEGATIVE'

test_get_predicted_class()
