from f00300_compile import *
import tensorflow as tf

def test_compile():
    model = tf.keras.models.Sequential()
    optimizer = tf.keras.optimizers.Adam()
    compile(optimizer)
    assert model.optimizer == optimizer

test_compile()
