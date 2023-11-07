from f00507_get_predicted_label import *
import tensorflow as tf


# Test case 1
logits = tf.constant([[0.1, 0.2, 0.7]])
id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
expected_label = 'label3'
assert get_predicted_label(logits, id2label) == expected_label

# Test case 2
logits = tf.constant([[0.7, 0.2, 0.1]])
id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
expected_label = 'label1'
assert get_predicted_label(logits, id2label) == expected_label

# Test case 3
logits = tf.constant([[0.3, 0.4, 0.3]])
id2label = {0: 'label1', 1: 'label2', 2: 'label3'}
expected_label = 'label2'
assert get_predicted_label(logits, id2label) == expected_label
