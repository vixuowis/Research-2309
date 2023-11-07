from f00806_run_model_with_xla import *
import tensorflow as tf

def test_run_model_with_xla():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    output = run_model_with_xla(model)

    assert output.shape == (1, 5)
    assert tf.reduce_sum(output) > 0
    assert tf.reduce_sum(output) < 5
