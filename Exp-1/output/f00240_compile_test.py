from f00240_compile import *
import tensorflow as tf

def test_compile():
    model = tf.keras.models.Sequential()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)

    # Test case 1
    assert isinstance(model, tf.keras.models.Sequential)
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

    # Test case 2
    assert model.optimizer.learning_rate == 0.001

    # Test case 3
    assert model.loss is None

    # Test case 4
    assert model.losses == []

    # Test case 5
    assert model.metrics == []

    print('All test cases pass')

test_compile()
