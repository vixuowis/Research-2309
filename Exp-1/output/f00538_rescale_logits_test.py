from f00538_rescale_logits import *
import tensorflow as tf
import numpy as np

# Test case 1
logits_1 = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
image_1 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
expected_1 = 1

# Test case 2
logits_2 = tf.constant([[[[5, 2], [3, 4]], [[5, 6], [7, 8]]]])
image_2 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
expected_2 = 0

# Test case 3
logits_3 = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
image_3 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
expected_3 = 1

# Test case 4
logits_4 = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
image_4 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
expected_4 = 1

# Test case 5
logits_5 = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
image_5 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
expected_5 = 1

# Run the tests
test_result_1 = rescale_logits(logits_1, image_1)
test_result_2 = rescale_logits(logits_2, image_2)
test_result_3 = rescale_logits(logits_3, image_3)
test_result_4 = rescale_logits(logits_4, image_4)
test_result_5 = rescale_logits(logits_5, image_5)

# Assert the results
np.testing.assert_equal(test_result_1, expected_1)
np.testing.assert_equal(test_result_2, expected_2)
np.testing.assert_equal(test_result_3, expected_3)
np.testing.assert_equal(test_result_4, expected_4)
np.testing.assert_equal(test_result_5, expected_5)
