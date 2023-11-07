from f00327_create_optimizer import *
def test_create_optimizer():
    learning_rate = 2e-5
    weight_decay_rate = 0.01
    optimizer = create_optimizer(learning_rate, weight_decay_rate)
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
    assert optimizer.learning_rate == learning_rate
    assert optimizer.weight_decay_rate == weight_decay_rate

test_create_optimizer()
