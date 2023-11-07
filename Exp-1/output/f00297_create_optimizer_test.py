from f00297_create_optimizer import *
def test_create_optimizer():
    learning_rate = 2e-5
    weight_decay_rate = 0.01
    optimizer = create_optimizer(learning_rate, weight_decay_rate)
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)

    learning_rate = 1e-4
    weight_decay_rate = 0.001
    epsilon = 1e-6
    optimizer = create_optimizer(learning_rate, weight_decay_rate, epsilon)
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)

test_create_optimizer()
