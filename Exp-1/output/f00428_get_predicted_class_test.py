from f00428_get_predicted_class import *
def test_get_predicted_class():
    logits = tf.constant([0.1, 0.5, 0.3])
    predicted_class = get_predicted_class(logits)
    assert predicted_class == 1

    logits = tf.constant([0.3, 0.1, 0.6])
    predicted_class = get_predicted_class(logits)
    assert predicted_class == 2

    logits = tf.constant([0.2, 0.7, 0.1])
    predicted_class = get_predicted_class(logits)
    assert predicted_class == 1

    logits = tf.constant([0.4, 0.4, 0.2])
    predicted_class = get_predicted_class(logits)
    assert predicted_class == 0

    logits = tf.constant([0.9, 0.05, 0.05])
    predicted_class = get_predicted_class(logits)
    assert predicted_class == 0
